import asyncio
import os
from contextlib import asynccontextmanager

import httpx
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from pydantic import ValidationError

from common.api import OCRResult, authorize, read_image


def backend_urls() -> list[str]:
    urls = [url.strip().rstrip("/") for url in os.getenv("MODEL_URLS", "").split(",") if url.strip()]
    if not urls:
        raise ValueError("MODEL_URLS must contain at least one model service URL")
    return urls


@asynccontextmanager
async def lifespan(app: FastAPI):
    urls = backend_urls()
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(float(os.getenv("MODEL_TIMEOUT_SECONDS", "120")), connect=5),
        trust_env=False,
    ) as client:
        app.state.client = client
        app.state.urls = urls
        app.state.next_backend = 0
        yield


app = FastAPI(title="PP-OCRv6 Router API", lifespan=lifespan)


def request_order() -> list[str]:
    urls = app.state.urls
    start = app.state.next_backend
    app.state.next_backend = (start + 1) % len(urls)
    return [urls[(start + offset) % len(urls)] for offset in range(len(urls))]


@app.get("/healthz")
async def health():
    responses = await asyncio.gather(
        *(app.state.client.get(f"{url}/healthz", timeout=3) for url in app.state.urls),
        return_exceptions=True,
    )
    ready = sum(isinstance(response, httpx.Response) and response.is_success for response in responses)
    if not ready:
        raise HTTPException(503, "No OCR model is available")
    return {"status": "ok", "ready_models": ready, "total_models": len(app.state.urls)}


@app.post("/v1/ocr", response_model=OCRResult, dependencies=[Depends(authorize)])
async def ocr(file: UploadFile = File(...)):
    data = await read_image(file)
    failures = []
    for url in request_order():
        try:
            response = await app.state.client.post(
                f"{url}/v1/ocr",
                files={"file": (file.filename or "image", data, file.content_type or "application/octet-stream")},
            )
            if response.status_code == 503:
                failures.append(url)
                continue
            if response.status_code in {400, 413, 415, 422}:
                raise HTTPException(response.status_code, response.json().get("detail", "Invalid image request"))
            response.raise_for_status()
            return OCRResult.model_validate(response.json())
        except HTTPException:
            raise
        except httpx.TimeoutException:
            failures.append(url)
        except (httpx.HTTPError, ValidationError, ValueError):
            failures.append(url)
    raise HTTPException(
        503,
        "All OCR model instances are busy or unavailable; retry later",
        headers={"Retry-After": "1"},
    )
