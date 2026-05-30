# segment-3gs

[![Powered by SpAItial AI](https://img.shields.io/badge/Powered%20by-SpAItial%20AI-7cc8ff?style=flat-square)](https://app.spaitial.ai)

**A Gaussian-splat segmentation playground, powered by [SpAItial AI](https://app.spaitial.ai)
worlds:** generate a 3D world on SpAItial AI, view its Gaussian splat in the browser, segment
objects in it on demand with [SAM3](https://huggingface.co/facebook/sam3), lift the 2D masks
into the 3D Gaussians, then recolor / isolate those Gaussians — "segment once, reuse forever".

[**SpAItial AI**](https://app.spaitial.ai) is a world-model company: describe or upload an
image and it generates a full 3D world you can explore and export. This demo is a showcase of
what you can build on top of those worlds — it loads a `.ply` Gaussian splat you export
straight from SpAItial AI.

This is a **starting point to fork**, not a product. It wires up a PlayCanvas splat viewer, a
local Python SAM server, and a full 2D→3D lift + multi-view voting + true gsplat recolor
pipeline so you can build your own splat-understanding tools on top.

<!-- screenshot: drop a screenshot at docs/screenshot.png and it renders below -->
<!-- ![segment-3gs](docs/screenshot.png) -->

## What this is

The scene is **static**, so segmentation is "segment once, reuse forever" rather than
per-frame. You point the camera (or run a batch sweep), the server runs a single SAM forward
on the rendered frame, and the client **lifts** each 2D mask onto the splats it covers and
stores the per-Gaussian index set. From then on the object lives in 3D — selectable,
re-framable, recolorable — with no further inference.

See **[`docs/ARCHITECTURE.md`](./docs/ARCHITECTURE.md)** for the full write-up: component +
sequence + lift + batch-sweep + recolor + coordinate diagrams, the `/segment` API contract,
the 3D lift algorithm, per-Gaussian voting merge, camera/coordinate spaces, persistence, and
model tiers.

## Quickstart

### Prerequisites

- **Node 18+** (recent LTS) and npm.
- **Python 3.11+** for the segmentation server (only needed for real masks — the viewer runs
  against a built-in mock without it).
- **macOS Apple Silicon (MPS)** is the reference target; CUDA and CPU also work. SAM is
  seconds-per-call on MPS, so segmentation here is **on-demand, not real-time**.
- A **Hugging Face account** with SAM3 license acceptance if you want the gated open-vocab
  `sam3` tier (see below). The `mock` and `mobile_sam` tiers need no HF account.

### 1. Install

```bash
npm install
```

### 2. Generate your world with SpAItial AI

The repo intentionally ships **without** any splat assets (they're heavy and gitignored). The
recommended way to get one is to **generate a world with [SpAItial AI](https://app.spaitial.ai)**
and export it:

1. **Create a world** at **[app.spaitial.ai](https://app.spaitial.ai)** — describe a scene or
   upload an image, and SpAItial AI generates a full 3D world (generation takes a few minutes).
2. **Download the world as a `.ply` Gaussian splat** — the app exports a ready-to-use,
   PlayCanvas-compatible `.ply` directly (no conversion needed) — and drop it at
   `public/world.ply`. The viewer derives the pivot, initial zoom, and spatial-grid bounds from
   the splat centers at runtime — **no per-scene config or manifest**.
3. **Export the world's simplified collision mesh** (optional) and drop it at
   `public/mesh_simplified.ply`. It is used **only at startup** to ground the initial camera on
   the floor. **If it's missing, the app still boots** and grounds the spawn on the splat
   bounds instead.
4. The splat path lives in one place — `SPLAT_URL` at the top of
   [`src/main.ts`](./src/main.ts) — change it for a different filename.

So the full workflow is: **create a world on SpAItial AI → export the splat `.ply` → export the
collision mesh → drop both into `public/`.**

> **Doing it programmatically?** The [SpAItial developer API](https://api.spaitial.ai)
> (`POST /v1/worlds`) creates worlds from text or image input; download the splat from
> `/v1/worlds/requests/:id/splat` and start the simplified-mesh export with
> `POST /v1/worlds/requests/:id/exports/mesh-simplified`. See **[About SpAItial AI](#about-spaitial-ai)**.

Only needed for the **developer API** path, which returns `.spz` (the app already gives you a
`.ply`). Convert a raw `.spz` to a PlayCanvas-compatible `.ply` with:

```bash
python3 -m venv .venv-spz
.venv-spz/bin/pip install "spz @ git+https://github.com/nianticlabs/spz" numpy
.venv-spz/bin/python scripts/convert-spz-to-playcanvas-ply.py <input>.spz public/world.ply --max-splats 0
```

### 3. Run the SAM server

Pick a tier (details + latencies in [`server/README.md`](./server/README.md)):

```bash
cd server

# Tier 1 — mock: deterministic ellipses, ZERO ML deps. Instant, great for frontend work.
SAM_MODEL=mock ./run.sh            # http://localhost:8765

# Tier 2 — mobile_sam / fallback: real masks, NO gated weights (MobileSAM points/boxes +
# FastSAM text via CLIP). Sub-second per segment on MPS.
SAM_MODEL=mobile_sam ./run.sh

# Tier 3 — sam3: gated, open-vocab "segment all <noun>" text segmentation.
#   1) Accept the license at https://huggingface.co/facebook/sam3
#   2) Authenticate, then run:
hf auth login                      # paste a Hugging Face token
SAM_MODEL=sam3 ./run.sh            # ~3.4 GB download on first run; ~13 s/call on MPS
```

`run.sh` creates `server/.venv`, installs `requirements.txt`, and launches uvicorn on
`:8765`. The viewer falls back to the built-in **mock** mask if the server is unreachable, so
`npm run dev` always shows *something*.

### 4. Run the viewer

```bash
npm run dev
```

Open the printed Vite URL.

## Features

- **Orbit / zoom / pan viewer** around a movable pivot (no physics, no Rapier).
- **Floor-grounded spawn** — robust percentile-trimmed splat bounds + an optional collision
  mesh place the initial camera at standing eye height.
- **Segment View** (`G`) — type a concept (e.g. `sofa`), press `G`, and the current frame is
  segmented and lifted into the splats. Concept-only (open-vocab text) — no point/click prompts.
- **Batch Segment** — a one-time **16-view look-around sweep** ("stand in the center and turn
  around") over an editable concept list, with a live progress bar, cost estimate, and cancel.
- **Confidence-voted multi-view merge** — masks of the same object from different views merge
  via **per-Gaussian voting** (vote counts + accumulated scores), pruning splats only one
  noisy view grabbed.
- **Multi-select** — toggle any number of objects on; each lights up in its own color.
- **3D point-cloud highlight + tracking labels** — selected objects render as an on-top point
  cloud with a DOM label that follows the object's centroid as the camera moves.
- **"Gaussian view" modes** — beyond the point overlay, recolor the *actual* Gaussians:
  **Point highlight** (overlay only), **Recolor + dim rest**, or **Isolate selected** (hide
  everything else).
- **localStorage persistence** — segmented objects, the selection, and the concept list all
  survive a page reload.

## How it works

1. **Render** a Gaussian splat in PlayCanvas; build a uniform spatial grid over the world-space
   splat centers.
2. **Capture** the rendered frame (downscaled to ≤1024 px JPEG via `OffscreenCanvas`) plus the
   camera pose at trigger time.
3. **Segment** — POST the frame + concept text to the local SAM server; get back instance masks.
4. **Lift** each mask into 3D: cull candidate splats to the mask's frustum slab via the grid,
   project them with the captured pose, keep those inside the mask within a depth band.
5. **Merge + vote** the lifted indices into the registry (per-Gaussian voting across views).
6. **Visualize** — point-cloud highlight + tracking label, and optionally recolor / isolate the
   underlying Gaussians via a `gsplatModifyVS` shader-chunk override and a per-splat id texture.

Full detail, diagrams, and the API contract: **[`docs/ARCHITECTURE.md`](./docs/ARCHITECTURE.md)**.

### Controls

| Input | Action |
|---|---|
| **Drag** | orbit (yaw/pitch) around the pivot |
| **Wheel** | zoom |
| **W A S D / arrows** | pan the pivot |
| **Q / E** | lower / raise the pivot |
| Type a concept + **G** | segment the current view (open-vocab text prompt) |
| **Batch Segment** | one-time 16-view look-around sweep over the concept list |
| **Gaussian view** select | point highlight / recolor + dim / isolate |

## Requirements & limitations

- **Not real-time.** SAM3 is ~13 s/call warm (~20 s+ cold) per concept on Apple MPS, so a full
  batch (`16 × concepts` calls) is a leave-it-running job (e.g. 8 concepts ≈ ~28 min on SAM3).
  This is why segmentation is on-demand and lifted-once, not a per-frame loop. `mobile_sam` is
  sub-second per segment.
- **SAM3 weights are gated** (Hugging Face license acceptance + `hf auth login`), ~3.4 GB.
- **`mobile_sam` text needs CLIP** weights (FastSAM + CLIP); on SSL-restricted networks the
  CLIP download can fail — point/box prompts still work without it.
- **"Segment everything" is unsupported by SAM3** (it requires a prompt); only FastSAM does it.
- **Main-thread lift.** The lift projects on the main thread; grid culling keeps it cheap, but a
  Web Worker is the natural next step.
- SAM3 **video/tracking is CUDA-only** and unused here.

## About SpAItial AI

**[SpAItial AI](https://app.spaitial.ai)** is a world-model company building foundation models
for 3D. From a text prompt or a single image, it generates a complete, explorable **3D world**
that you can export as a **`.ply` Gaussian splat** (and a **simplified collision mesh**) —
exactly the assets this demo loads. The worlds you see here are generated and exported with
SpAItial AI.

- **Web app — create & explore worlds:** [app.spaitial.ai](https://app.spaitial.ai)
- **Developer API — generate worlds programmatically:** [api.spaitial.ai](https://api.spaitial.ai)
  — `POST /v1/worlds` (text / image / panorama input), poll to completion, download the splat,
  and export a full or simplified mesh. The API returns splats as `.spz`; convert to a
  PlayCanvas `.ply` with the script above. (Developer keys + usage at
  [developers.spaitial.ai](https://developers.spaitial.ai).)

This segment-3gs is an open-source showcase of what you can build on top of SpAItial AI
worlds — bring a world, segment it, and lift the masks into 3D.

## License & attribution

This project's code is **[MIT](./LICENSE)** licensed — fork it freely.

**Important:** the SAM3 **model weights** are *not* MIT. They are distributed under
[Meta's SAM license](https://huggingface.co/facebook/sam3) and are gated on Hugging Face — you
must accept that license yourself to download and use them. This repo ships no weights.

Credits:

- **[SpAItial AI](https://app.spaitial.ai)** — 3D world generation; the Gaussian-splat worlds
  (`world.ply`) and collision meshes (`mesh_simplified.ply`) this demo loads are created and
  exported with SpAItial AI. This project exists to showcase those worlds.
- **[PlayCanvas](https://playcanvas.com/)** — WebGL engine + Gaussian-splat rendering.
- **[Ultralytics](https://github.com/ultralytics/ultralytics)** — MobileSAM / FastSAM (the
  no-gated-weights `fallback` tier).
- **[Meta AI — Segment Anything 3](https://huggingface.co/facebook/sam3)** — open-vocabulary
  concept segmentation (the gated `sam3` tier).
