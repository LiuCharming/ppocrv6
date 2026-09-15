import io
import threading

import numpy as np
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from PIL import Image

from common import api
from model_service import main as model


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


def test_gpu_capacity(monkeypatch):
    started, release = threading.Event(), threading.Event()

    class SlowEngine(Engine):
        def predict(self, image):
            started.set()
            assert release.wait(5)
            return super().predict(image)

    monkeypatch.setattr(model, "create_engine", SlowEngine)
    with TestClient(model.app) as client:
        responses = []
        worker = threading.Thread(target=lambda: responses.append(
            client.post("/v1/ocr", headers=AUTH, files={"file": png()})))
        worker.start()
        try:
            assert started.wait(5)
            assert client.get("/healthz").status_code == 200
            assert client.post("/v1/ocr", headers=AUTH, files={"file": png()}).status_code == 503
        finally:
            release.set()
            worker.join(5)
        assert responses[0].status_code == 200
