"""Smoke tests for the segmentation server.

Run from the `server/` dir:
    python -m pytest test_smoke.py -v
or directly:
    python test_smoke.py

All tests run in SAM_MODEL=mock so there are no ML deps / network downloads.
"""

from __future__ import annotations

import base64
import io
import os

os.environ.setdefault("SAM_MODEL", "mock")  # must be set before importing app

import numpy as np
from PIL import Image

from segmenters import MockSegmenter, Point, Prompts, build_segmenter

WIDTH, HEIGHT = 320, 240


def _synthetic_image(width: int = WIDTH, height: int = HEIGHT) -> Image.Image:
    """A deterministic gradient + block image (no external assets)."""
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    arr[..., 0] = np.linspace(0, 255, width, dtype=np.uint8)[None, :]
    arr[..., 1] = np.linspace(0, 255, height, dtype=np.uint8)[:, None]
    arr[height // 4 : 3 * height // 4, width // 4 : 3 * width // 4, 2] = 200
    return Image.fromarray(arr, mode="RGB")


def _image_b64(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# --------------------------------------------------------------------------- #
# Segmenter-class-level tests (mock mode, no server)
# --------------------------------------------------------------------------- #
def test_factory_returns_mock():
    seg = build_segmenter("mock")
    assert isinstance(seg, MockSegmenter)
    assert seg.text_supported is True


def test_mock_text_prompt_shape():
    seg = build_segmenter("mock")
    img = _synthetic_image()
    res = seg.segment(img, Prompts(text=["sofa", "teacup"]), WIDTH, HEIGHT)
    assert len(res.masks) == 2
    labels = [m.label for m in res.masks]
    assert labels == ["sofa", "teacup"]
    for m in res.masks:
        assert m.mask.shape == (HEIGHT, WIDTH)
        assert m.mask.dtype == bool or m.mask.dtype == np.bool_
        assert 0.0 <= m.score <= 1.0
        x0, y0, x1, y1 = m.bbox
        assert x1 > x0 and y1 > y0  # non-empty extent


def test_mock_point_prompt():
    seg = build_segmenter("mock")
    img = _synthetic_image()
    pts = [Point(x=160, y=120, label=1), Point(x=10, y=10, label=0)]
    res = seg.segment(img, Prompts(points=pts), WIDTH, HEIGHT)
    # background point (label 0) is skipped -> only 1 mask
    assert len(res.masks) == 1
    assert res.masks[0].label == "point_0"


def test_mock_box_prompt():
    seg = build_segmenter("mock")
    res = seg.segment(_synthetic_image(), Prompts(boxes=[[50, 50, 150, 150]]), WIDTH, HEIGHT)
    assert len(res.masks) == 1
    assert res.masks[0].label == "object_0"


def test_mock_no_prompts_segments_centered():
    seg = build_segmenter("mock")
    res = seg.segment(_synthetic_image(), Prompts(), WIDTH, HEIGHT)
    assert len(res.masks) == 1
    assert res.masks[0].label == "object_0"


# --------------------------------------------------------------------------- #
# FastAPI app tests via TestClient (mock mode)
# --------------------------------------------------------------------------- #
def _client():
    from fastapi.testclient import TestClient

    import app as app_module

    return TestClient(app_module.app)


def test_health_endpoint():
    client = _client()
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model"] == "mock"
    assert body["device"] in ("mps", "cpu", "cuda")
    assert body["text_supported"] is True


def test_segment_endpoint_text():
    client = _client()
    payload = {
        "image": _image_b64(_synthetic_image()),
        "width": WIDTH,
        "height": HEIGHT,
        "prompts": {"text": ["sofa"]},
        "multimask": False,
    }
    r = client.post("/segment", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["width"] == WIDTH and body["height"] == HEIGHT
    assert body["model"] == "mock"
    assert isinstance(body["elapsed_ms"], (int, float))
    assert len(body["masks"]) == 1
    m = body["masks"][0]
    assert m["label"] == "sofa"
    assert len(m["bbox"]) == 4

    # mask_png must decode to a single-channel image at the requested resolution
    raw = base64.b64decode(m["mask_png"])
    decoded = Image.open(io.BytesIO(raw))
    assert decoded.size == (WIDTH, HEIGHT)
    assert decoded.mode == "L"
    vals = set(np.unique(np.asarray(decoded)).tolist())
    assert vals.issubset({0, 255})  # strictly binary


def test_segment_endpoint_data_uri_prefix():
    client = _client()
    payload = {
        "image": "data:image/png;base64," + _image_b64(_synthetic_image()),
        "width": WIDTH,
        "height": HEIGHT,
        "prompts": {"points": [{"x": 160, "y": 120, "label": 1}]},
    }
    r = client.post("/segment", json=payload)
    assert r.status_code == 200, r.text
    assert len(r.json()["masks"]) == 1


def test_segment_endpoint_bad_image():
    client = _client()
    payload = {"image": "not-base64-@@@", "width": WIDTH, "height": HEIGHT}
    r = client.post("/segment", json=payload)
    assert r.status_code == 400


if __name__ == "__main__":
    import sys
    import traceback

    failures = 0
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception:
            failures += 1
            print(f"FAIL {t.__name__}")
            traceback.print_exc()
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
