import logging
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

import asyncio
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile

from common.api import OCRResult, TextLine, authorize, decode_image, read_image

logger = logging.getLogger(__name__)
MODEL_SIZE = os.getenv("MODEL_SIZE", "medium")
if MODEL_SIZE not in {"medium", "small", "tiny"}:
    raise ValueError("MODEL_SIZE must be medium, small or tiny")
MODEL = f"PP-OCRv6_{MODEL_SIZE}"


def create_engine():
    import paddle
    from paddleocr import PaddleOCR

    if not paddle.is_compiled_with_cuda() or paddle.device.cuda.device_count() < 1:
        raise RuntimeError("CUDA GPU unavailable; check NVIDIA driver and Container Toolkit")
    return PaddleOCR(
        device="gpu:0",
        ocr_version="PP-OCRv6",
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize and call Paddle on the same single thread; one model copy per GPU.
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        app.state.executor = executor
        app.state.engine = await asyncio.get_running_loop().run_in_executor(executor, create_engine)
        app.state.busy = False
        yield
    finally:
        executor.shutdown(wait=True)


app = FastAPI(title="PP-OCRv6 GPU Model API", lifespan=lifespan)


@app.get("/healthz")
def health():
    return {"status": "ok", "model": MODEL, "device": "gpu:0"}


@app.post("/v1/ocr", response_model=OCRResult, dependencies=[Depends(authorize)])
async def ocr(file: UploadFile = File(...)):
    data = await read_image(file)
    if app.state.busy:
        raise HTTPException(503, "GPU busy; retry later", headers={"Retry-After": "1"})
    app.state.busy = True
    future = asyncio.get_running_loop().run_in_executor(
        app.state.executor, recognize, app.state.engine, data)
    # Client cancellation must not release GPU capacity while inference is running.
    future.add_done_callback(lambda _: setattr(app.state, "busy", False))
    try:
        return await asyncio.shield(future)
    except HTTPException:
        raise
    except Exception:
        logger.exception("OCR inference failed")
        raise HTTPException(500, "OCR inference failed") from None
