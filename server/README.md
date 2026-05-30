# segmen inference server

Local Python HTTP server that segments objects in a single rendered frame given
**text** and/or **click/box** prompts. Built for the PlayCanvas Gaussian-splat
web app frontend. On-demand only (seconds per call on Apple Silicon — not live FPS).

- **Primary model:** Meta **SAM3** (open-vocab "segment all instances of `<noun>`"
  like `sofa`, `teacup`, plus box/point prompts). Weights are **gated**.
- **Fallback (default-able):** ultralytics **MobileSAM/FastSAM** (click/box always,
  text via FastSAM+CLIP). No gated weights.
- **Mock:** deterministic ellipse masks, zero ML deps — used for smoke tests and
  so the frontend has something to hit immediately.

Backend is chosen with the `SAM_MODEL` env var:
`sam3` | `sam3.1` | `mobile_sam` | `sam2` | `fastsam` | `fallback` | `mock` (default).

---

## Quick start

```bash
cd server
SAM_MODEL=mock ./run.sh           # boots on http://localhost:8765, no ML deps needed
```

`run.sh` creates `server/.venv`, installs `requirements.txt`, exports
`PYTORCH_ENABLE_MPS_FALLBACK=1`, and runs uvicorn on port 8765.

Health check + a mock segmentation:

```bash
curl -s http://localhost:8765/health
# {"status":"ok","model":"mock","device":"mps","text_supported":true}
```

### Run the fallback (no gated weights, click/box + FastSAM text)

```bash
pip install ultralytics torch          # into server/.venv
SAM_MODEL=mobile_sam ./run.sh
```

ultralytics auto-downloads `mobile_sam.pt` (points/boxes) and `FastSAM-s.pt`
(text/everything) on first request. Device is MPS on Apple Silicon.

---

## Enabling real SAM3 (gated weights) — exact steps

1. **Accept the license** on the model page (Meta SAM License, requires HF login):
   - https://huggingface.co/facebook/sam3  (or `facebook/sam3.1`)
2. **Install the SAM3 deps** into the venv (a recent Transformers exposes the
   `Sam3Model` / `Sam3Processor` classes):
   ```bash
   cd server && source .venv/bin/activate
   pip install -U "transformers>=5.9" huggingface_hub accelerate torch pillow
   ```
3. **Authenticate** so the gated weights can download (~3.4 GB, F32):
   ```bash
   hf auth login          # or: huggingface-cli login   (paste a HF token)
   ```
4. **Run with SAM3:**
   ```bash
   SAM_MODEL=sam3 ./run.sh        # use sam3.1 for the newer checkpoint
   ```

Notes / limitations of the SAM3 path on this Mac:
- Image inference works on **MPS** but is **seconds per image** (~6.5 s/img on an
  M2 per the upstream MPS PR). The **video/tracking path is CUDA-only** and not used here.
- The 🤗 Transformers SAM3 *image* path supports **text + box** prompts. It has **no
  native point prompt**, so this server converts each foreground click into a small
  box (±16 px) around the point. Text + box prompts are passed through natively.
- If weights aren't accepted / not logged in / Transformers too old, `/segment`
  returns **HTTP 503** with an actionable message instead of crashing the server.

---

## API contract

Server listens on `http://localhost:8765`. CORS is enabled for
`http://localhost:5173` (and `*` for local dev).

### `GET /health`

```json
{ "status": "ok", "model": "mock", "device": "mps", "text_supported": true }
```

### `POST /segment`  (`Content-Type: application/json`)

```json
{
  "image": "<base64 PNG/JPEG, may include a data: URI prefix — it is stripped>",
  "width": 1280,
  "height": 720,
  "prompts": {
    "text": ["sofa", "teacup"],
    "points": [{ "x": 640, "y": 360, "label": 1 }],
    "boxes": [[100, 150, 500, 450]]
  },
  "multimask": false
}
```

- `prompts`, and each field within it, are optional. Pixel coordinates.
- `label` on a point: `1` = foreground, `0` = background.
- If **no prompts** are given: the fallback/FastSAM backend does "segment
  everything" (capped at 30 masks); SAM3 returns an empty `masks` array with a
  `message` (its image path requires a prompt); mock returns one centered mask.

**Response `200`:**

```json
{
  "masks": [
    {
      "label": "sofa",
      "score": 0.93,
      "mask_png": "<base64 PNG, single channel (L), values 0 or 255, width×height>",
      "bbox": [120, 200, 640, 540]
    }
  ],
  "width": 1280,
  "height": 720,
  "model": "sam3",
  "elapsed_ms": 5421.3,
  "message": null
}
```

- One mask object **per detected instance** (text prompts can yield many) and
  per point/box. Masks are always returned at the requested `width`×`height`.
- `label` is the text phrase for text prompts, else `point_<i>` / `object_<i>`.
- Errors: `400` for bad/undecodable image or base64; `503` for an actionable
  model/weight load failure (e.g. gated SAM3 not authenticated); `500` otherwise.

### Example `curl`

```bash
# build a tiny base64 PNG and segment "sofa" in mock mode
python3 - <<'PY'
import base64, io, json, urllib.request
from PIL import Image
buf = io.BytesIO(); Image.new("RGB", (320, 240), (40, 80, 160)).save(buf, "PNG")
payload = {"image": base64.b64encode(buf.getvalue()).decode(),
           "width": 320, "height": 240, "prompts": {"text": ["sofa"]}}
req = urllib.request.Request("http://localhost:8765/segment",
        data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
print(json.load(urllib.request.urlopen(req)).keys())
PY
```

---

## Tests

```bash
cd server && source .venv/bin/activate
python -m pytest test_smoke.py -v        # or: python test_smoke.py
```

Tests run in `SAM_MODEL=mock` (no ML deps, no downloads): they exercise the
segmenter classes directly and the FastAPI app via `TestClient` for `/health`
and `/segment`, asserting the response shape and that `mask_png` decodes to a
single-channel image at the requested resolution.

---

## Files

| file              | purpose                                                        |
|-------------------|----------------------------------------------------------------|
| `app.py`          | FastAPI app, schema, base64/image helpers, the two endpoints   |
| `segmenters.py`   | `Segmenter` ABC + `Sam3Segmenter` / `FallbackSegmenter` / `MockSegmenter`, device detection, factory |
| `requirements.txt`| pinned-ish deps (SAM3 gated deps commented with install notes) |
| `run.sh`          | venv bootstrap + install + uvicorn on :8765 (MPS fallback on)  |
| `test_smoke.py`   | synthetic-image smoke tests (mock mode)                        |
