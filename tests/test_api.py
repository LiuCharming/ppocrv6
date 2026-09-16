import io
import threading
import time
import asyncio

import httpx
import numpy as np
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from PIL import Image

from common import api
from model_service import main as model
from router_service import main as router


def png():
    output = io.BytesIO()
    Image.new("RGB", (8, 5), (255, 0, 0)).save(output, format="PNG")
    return output.getvalue()


class Engine:
    def predict(self, image):
        assert image[0, 0].tolist() == [0, 0, 255]
        return [{"rec_texts": ["你好", "OCR"], "rec_scores": np.array([0.99, 0.95]),
                 "rec_polys": np.array([[[0, 0], [4, 0], [4, 2], [0, 2]],
                                         [[0, 2], [4, 2], [4, 4], [0, 4]]])}]


@pytest.fixture(autouse=True)
def settings(monkeypatch):
    monkeypatch.setenv("API_KEY", "test-key")
    monkeypatch.setenv("OCR_CONCURRENCY", "2")
    monkeypatch.setenv("OCR_QUEUE_SIZE", "2")
    monkeypatch.setenv("MODEL_URLS", "http://model-a,http://model-b,http://model-c")
    monkeypatch.setattr(model, "create_engine", Engine)


AUTH = {"Authorization": "Bearer test-key"}


def test_model_output_and_validation(monkeypatch):
    with TestClient(model.app) as client:
        assert client.get("/healthz").status_code == 200
        assert client.post("/v1/ocr", files={"file": png()}).status_code == 401
        response = client.post("/v1/ocr", headers=AUTH, files={"file": png()})
        assert response.status_code == 200
        assert response.json()["text"] == "你好\nOCR"
        assert response.json()["width"] == 8
        assert response.json()["lines"][0]["polygon"][1] == [4.0, 0.0]
        assert client.post("/v1/ocr", headers=AUTH, files={"file": b"bad"}).status_code == 400
        assert client.post("/v1/ocr", headers=AUTH, files={"file": b""}).status_code == 400
        monkeypatch.setattr(api, "MAX_BYTES", 4)
        assert client.post("/v1/ocr", headers=AUTH, files={"file": png()}).status_code == 413


def test_empty_result():
    class Empty:
        def predict(self, image):
            return [{"rec_texts": [], "rec_scores": [], "rec_polys": []}]
    result = model.recognize(Empty(), png())
    assert result.text == "" and result.lines == []


def test_pixel_limit(monkeypatch):
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 30)
    with pytest.raises(HTTPException) as error:
        api.decode_image(png())
    assert error.value.status_code == 413


def test_concurrency_queue_and_capacity(monkeypatch):
    started, release = threading.Event(), threading.Event()
    monkeypatch.setenv("OCR_CONCURRENCY", "1")
    monkeypatch.setenv("OCR_QUEUE_SIZE", "1")

    class SlowEngine(Engine):
        def predict(self, image):
            started.set()
            assert release.wait(5)
            return super().predict(image)

    monkeypatch.setattr(model, "create_engine", SlowEngine)
    with TestClient(model.app) as client:
        responses = []
        first = threading.Thread(target=lambda: responses.append(
            client.post("/v1/ocr", headers=AUTH, files={"file": png()})))
        first.start()
        try:
            assert started.wait(5)
            assert client.get("/healthz").status_code == 200
            second = threading.Thread(target=lambda: responses.append(
                client.post("/v1/ocr", headers=AUTH, files={"file": png()})))
            second.start()
            deadline = time.monotonic() + 2
            while client.get("/healthz").json()["queued"] != 1:
                assert time.monotonic() < deadline
                time.sleep(0.01)
            # One inference runs and one waits; the third request exceeds capacity.
            assert client.post("/v1/ocr", headers=AUTH, files={"file": png()}).status_code == 503
        finally:
            release.set()
            first.join(5)
            second.join(5)
        assert [response.status_code for response in responses] == [200, 200]


def test_router_round_robin_and_failover():
    calls = []

    def handler(request):
        calls.append(str(request.url))
        if request.url.host == "model-b":
            return httpx.Response(503)
        if request.url.path == "/healthz":
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(200, json={
            "text": request.url.host,
            "lines": [],
            "model": "PP-OCRv6_medium",
            "width": 8,
            "height": 5,
        })

    with TestClient(router.app) as client:
        original = router.app.state.client
        upstream = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        router.app.state.client = upstream
        try:
            assert client.get("/healthz").json()["ready_models"] == 2
            first = client.post("/v1/ocr", headers=AUTH, files={"file": png()})
            second = client.post("/v1/ocr", headers=AUTH, files={"file": png()})
            assert first.json()["text"] == "model-a"
            assert second.json()["text"] == "model-c"
        finally:
            asyncio.run(upstream.aclose())
            router.app.state.client = original
