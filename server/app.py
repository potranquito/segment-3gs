"""FastAPI inference server for on-demand object segmentation.

Endpoints:
    GET  /health   -> backend / device / capability info
    POST /segment  -> run segmentation on a single base64 frame

Backend is selected with the ``SAM_MODEL`` env var (default ``mock``):
    sam3 | sam3.1 | mobile_sam | sam2 | fastsam | fallback | mock

Run:
    SAM_MODEL=mock uvicorn app:app --host 0.0.0.0 --port 8765
"""

from __future__ import annotations

import base64
import binascii
import io
import logging
import os
import time
from functools import lru_cache
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel, Field

from segmenters import (
    MaskResult,
    Point,
    Prompts,
    Segmenter,
    build_segmenter,
    detect_device,
)

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("server")

SAM_MODEL = os.environ.get("SAM_MODEL", "mock")
PORT = int(os.environ.get("PORT", "8765"))


# --------------------------------------------------------------------------- #
# Request / response schema (matches the frontend contract exactly)
# --------------------------------------------------------------------------- #
class PointModel(BaseModel):
    x: float
    y: float
    label: int = 1  # 1 = foreground, 0 = background


class PromptsModel(BaseModel):
    text: Optional[list[str]] = None
    points: Optional[list[PointModel]] = None
    boxes: Optional[list[list[float]]] = None  # [x0, y0, x1, y1]


class SegmentRequest(BaseModel):
    image: str = Field(..., description="base64 PNG/JPEG, optional data: URI prefix")
    width: int
    height: int
    prompts: Optional[PromptsModel] = None
    multimask: bool = False


class MaskModel(BaseModel):
    label: str
    score: float
    mask_png: str
    bbox: list[int]


class SegmentResponse(BaseModel):
    masks: list[MaskModel]
    width: int
    height: int
    model: str
    elapsed_ms: float
    message: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    model: str
    device: str
    text_supported: bool


# --------------------------------------------------------------------------- #
# base64 / image helpers
# --------------------------------------------------------------------------- #
def decode_image(b64: str) -> Image.Image:
    """Decode a (possibly data-URI prefixed) base64 PNG/JPEG into a PIL RGB image."""
    if not b64:
        raise HTTPException(status_code=400, detail="image is empty")
    if b64.startswith("data:"):
        # strip "data:image/png;base64," style prefix
        comma = b64.find(",")
        if comma != -1:
            b64 = b64[comma + 1 :]
    try:
        raw = base64.b64decode(b64, validate=False)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"invalid base64 image: {exc}") from exc
    try:
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:  # not a valid image
        raise HTTPException(status_code=400, detail=f"could not decode image: {exc}") from exc


def encode_mask_png(mask: np.ndarray, width: int, height: int) -> str:
    """Encode a boolean/uint8 mask as a single-channel (L) PNG, 0 or 255, base64."""
    arr = (np.asarray(mask).astype(bool).astype(np.uint8)) * 255
    img = Image.fromarray(arr, mode="L")
    if img.size != (width, height):
        img = img.resize((width, height), Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def to_prompts(p: Optional[PromptsModel]) -> Prompts:
    if p is None:
        return Prompts()
    return Prompts(
        text=[t for t in (p.text or []) if t and t.strip()],
        points=[Point(x=pt.x, y=pt.y, label=pt.label) for pt in (p.points or [])],
        boxes=[list(b) for b in (p.boxes or []) if len(b) == 4],
    )


# --------------------------------------------------------------------------- #
# Segmenter singleton (lazy model load on first request)
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def get_segmenter() -> Segmenter:
    seg = build_segmenter(SAM_MODEL, device=detect_device())
    logger.info("Configured backend '%s' (device=%s, requested SAM_MODEL=%s)", seg.name, seg.device, SAM_MODEL)
    return seg


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
app = FastAPI(title="segmen inference server", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    seg = get_segmenter()
    return HealthResponse(
        status="ok",
        model=SAM_MODEL,
        device=seg.device,
        text_supported=seg.text_supported,
    )


@app.post("/segment", response_model=SegmentResponse)
def segment(req: SegmentRequest) -> SegmentResponse:
    t0 = time.perf_counter()
    image = decode_image(req.image)
    prompts = to_prompts(req.prompts)
    seg = get_segmenter()

    logger.info(
        "segment: model=%s text=%d points=%d boxes=%d size=%dx%d",
        seg.name,
        len(prompts.text),
        len(prompts.points),
        len(prompts.boxes),
        req.width,
        req.height,
    )

    try:
        result = seg.segment(image, prompts, req.width, req.height, multimask=req.multimask)
    except RuntimeError as exc:
        # actionable model/weight errors -> 503 so the server stays up
        logger.error("segmentation backend error: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # unexpected
        logger.exception("unexpected segmentation error")
        raise HTTPException(status_code=500, detail=f"segmentation failed: {exc}") from exc

    masks: list[MaskModel] = []
    for m in result.masks:  # type: MaskResult
        masks.append(
            MaskModel(
                label=m.label,
                score=round(float(m.score), 4),
                mask_png=encode_mask_png(m.mask, req.width, req.height),
                bbox=[int(v) for v in m.bbox],
            )
        )

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return SegmentResponse(
        masks=masks,
        width=req.width,
        height=req.height,
        model=SAM_MODEL,
        elapsed_ms=round(elapsed_ms, 1),
        message=result.message,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT)
