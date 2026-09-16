import logging
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass

import asyncio
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile

from common.api import OCRResult, TextLine, authorize, decode_image, read_image

logger = logging.getLogger(__name__)
MODEL_SIZE = os.getenv("MODEL_SIZE", "medium")
if MODEL_SIZE not in {"medium", "small", "tiny"}:
    raise ValueError("MODEL_SIZE must be medium, small or tiny")
MODEL = f"PP-OCRv6_{MODEL_SIZE}"
DEVICE = os.getenv("OCR_DEVICE", "cpu")
if DEVICE != "cpu" and DEVICE != "gpu:0":
    raise ValueError("OCR_DEVICE must be cpu or gpu:0")


def positive_int_env(name: str, default: int, minimum: int = 1) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError as error:
        raise ValueError(f"{name} must be an integer") from error
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


@dataclass
class OCRJob:
    data: bytes
    future: asyncio.Future[OCRResult]


def create_engine():
    import paddle
    from paddleocr import PaddleOCR

    if DEVICE == "gpu:0" and (not paddle.is_compiled_with_cuda() or paddle.device.cuda.device_count() < 1):
        raise RuntimeError("CUDA GPU unavailable; check NVIDIA driver and Container Toolkit")
    return PaddleOCR(
        device=DEVICE,
        text_detection_model_name=f"{MODEL}_det",
        text_recognition_model_name=f"{MODEL}_rec",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )


def recognize(engine, data: bytes) -> OCRResult:
    import numpy as np

    image = decode_image(data)
    # PaddleOCR's ndarray input uses OpenCV BGR channel order.
    pixels = np.asarray(image)[:, :, ::-1].copy()
    lines = []
    for result in engine.predict(pixels):
        texts = result["rec_texts"]
        scores = result["rec_scores"]
        polygons = result["rec_polys"]
        if not (len(texts) == len(scores) == len(polygons)):
            raise RuntimeError("Unexpected OCR result lengths")
        for text, score, polygon in zip(texts, scores, polygons):
            lines.append(TextLine(text=text, confidence=float(score),
                                  polygon=[[float(x), float(y)] for x, y in polygon]))
    return OCRResult(text="\n".join(line.text for line in lines), lines=lines,
                     model=MODEL, width=image.width, height=image.height)


async def worker(queue: asyncio.Queue[OCRJob], executor: ThreadPoolExecutor, engine) -> None:
    """Keep each PaddleOCR instance on its own dedicated OS thread."""
    loop = asyncio.get_running_loop()
    while True:
        job = await queue.get()
        try:
            result = await loop.run_in_executor(executor, recognize, engine, job.data)
            if not job.future.done():
                job.future.set_result(result)
        except Exception as error:
            if not job.future.done():
                job.future.set_exception(error)
        finally:
            queue.task_done()


@asynccontextmanager
async def lifespan(app: FastAPI):
    concurrency = positive_int_env("OCR_CONCURRENCY", 2)
    queue_size = positive_int_env("OCR_QUEUE_SIZE", 16)
    queue: asyncio.Queue[OCRJob] = asyncio.Queue(maxsize=queue_size)
    executors: list[ThreadPoolExecutor] = []
    tasks: list[asyncio.Task] = []
    try:
        # Every worker has an independent PaddleOCR instance and a pinned thread.
        # This avoids sharing a Predictor between simultaneous inferences.
        loop = asyncio.get_running_loop()
        for _ in range(concurrency):
            executor = ThreadPoolExecutor(max_workers=1)
            executors.append(executor)
            engine = await loop.run_in_executor(executor, create_engine)
            tasks.append(asyncio.create_task(worker(queue, executor, engine)))
        app.state.queue = queue
        app.state.concurrency = concurrency
        app.state.queue_size = queue_size
        yield
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        for executor in executors:
            executor.shutdown(wait=True)


app = FastAPI(title="PP-OCRv6 Model API", lifespan=lifespan)


@app.get("/healthz")
def health():
    return {
        "status": "ok",
        "model": MODEL,
        "device": DEVICE,
        "concurrency": app.state.concurrency,
        "queued": app.state.queue.qsize(),
        "queue_capacity": app.state.queue_size,
    }


@app.post("/v1/ocr", response_model=OCRResult, dependencies=[Depends(authorize)])
async def ocr(file: UploadFile = File(...)):
    data = await read_image(file)
    future: asyncio.Future[OCRResult] = asyncio.get_running_loop().create_future()
    try:
        app.state.queue.put_nowait(OCRJob(data=data, future=future))
    except asyncio.QueueFull:
        raise HTTPException(
            503,
            "OCR queue is full; retry later",
            headers={"Retry-After": "1"},
        ) from None
    try:
        # Client cancellation does not remove a job already accepted by the service.
        return await asyncio.shield(future)
    except HTTPException:
        raise
    except Exception:
        logger.exception("OCR inference failed")
        raise HTTPException(500, "OCR inference failed") from None
