import io
import os
import secrets
import warnings

from fastapi import Header, HTTPException, UploadFile
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, Field

MAX_BYTES = int(os.getenv("MAX_IMAGE_MB", "10")) * 1024 * 1024
Image.MAX_IMAGE_PIXELS = int(os.getenv("MAX_IMAGE_PIXELS", "25000000"))


class TextLine(BaseModel):
    text: str
    confidence: float = Field(ge=0, le=1)
    polygon: list[list[float]]


class OCRResult(BaseModel):
    text: str
    lines: list[TextLine]
    model: str
    width: int
    height: int


def authorize(authorization: str | None = Header(default=None)):
    key = os.getenv("API_KEY", "")
    if key and not secrets.compare_digest(authorization or "", f"Bearer {key}"):
        raise HTTPException(401, "Invalid API key")


async def read_image(file: UploadFile) -> bytes:
    try:
        data = await file.read(MAX_BYTES + 1)
    finally:
        await file.close()
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Image exceeds upload limit")
    if not data:
        raise HTTPException(400, "Empty image")
    return data


def decode_image(data: bytes) -> Image.Image:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as source:
                if source.format not in {"PNG", "JPEG", "WEBP", "BMP", "TIFF"}:
                    raise HTTPException(415, "Supported formats: PNG, JPEG, WEBP, BMP, TIFF")
                if getattr(source, "n_frames", 1) != 1:
                    raise HTTPException(400, "Only single-frame images are supported")
                if source.width * source.height > Image.MAX_IMAGE_PIXELS:
                    raise HTTPException(413, "Image exceeds pixel limit")
                return ImageOps.exif_transpose(source).convert("RGB")
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise HTTPException(413, "Image exceeds pixel limit") from None
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(400, "Invalid or damaged image") from None
