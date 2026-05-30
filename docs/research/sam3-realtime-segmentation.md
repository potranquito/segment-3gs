# Real-Time SAM3 Segmentation for the Moonbase Greenhouse Splat World

**Research report — May 2026. No code was changed.**

Goal: segment objects (sofa, teacup, chairs, pillows) in a PlayCanvas 3D Gaussian Splat scene *as the camera moves*, by feeding live rendered frames to Meta's SAM3 running locally on Apple Silicon.

**TL;DR up front:** SAM3 is real and excellent at exactly your task (open-vocabulary "segment by concept name"), but it is a ~0.86B-param model whose video/tracking path is **CUDA-only** and whose image path on Apple Silicon MPS runs in **seconds, not milliseconds**. True per-frame live streaming of SAM3 on a Mac alone is not real-time. For a *static* splat world the right architecture is **offline lift-to-3D once, then segmentation is free at render time** — that beats live 2D inference on every axis. Live streaming only makes sense if you add a CUDA GPU box. Details and 3 ranked options below.

---

## 1. SAM3 (Segment Anything Model 3)

**Status (May 2026): released and actively maintained.**

- Repo: `https://github.com/facebookresearch/sam3` (~10.2k stars, Python, last push May 2026).
- Project page: `https://ai.meta.com/sam3/`
- Paper: *SAM 3: Segment Anything with Concepts*, Carion et al., arXiv **2511.16719**.
- Weights (gated, manual approval): `https://huggingface.co/facebook/sam3` and the newer `https://huggingface.co/facebook/sam3.1`. You must request access and `hf auth login`.
- License: **custom "SAM License"** (`NOASSERTION` on GitHub). Permits commercial use; prohibits military/warfare and reverse engineering. Read the LICENSE before shipping a product.
- **SAM 3.1** released **2026-03-27** ("Object Multiplex"): shared-memory joint multi-object tracking, significantly faster without accuracy loss. Needs the latest repo code.

### What's new vs SAM2 — this is the headline feature for you

SAM2 only does **Promptable Visual Segmentation (PVS)**: you give points/boxes/masks and it segments **one** object. SAM3 adds **Promptable Concept Segmentation (PCS)**: give it a **short noun phrase** ("sofa", "teacup", "striped pillow") and/or image exemplars, and it returns instance masks + IDs for **every matching instance** in the image/video. This is precisely your "segment by concept name" requirement — no manual clicking per object.

- Architecture: a **DETR-based detector** + a **SAM2-style memory tracker**, both sharing one **Perception Encoder (PE)** vision-language backbone. A novel "presence head" decouples recognition from localization. (paper §3)
- Text prompts are constrained to **simple noun phrases** (noun + modifiers), not long referring expressions. For complex queries you wrap it with an MLLM ("SAM 3 Agent"). So "teacup" works; "the cup the astronaut drank from" does not directly.
- Accuracy: zero-shot **mask AP 48.8 on LVIS** (vs 38.5 prior SOTA); **2×+** the cgF1 of best baselines on the new SA-Co benchmark (270K concepts, 50× more than prior); also **beats SAM2 on PVS/VOS** (e.g. MOSEv2 +6.5). Reaches ~75–80% of human performance on SA-Co.

### Sizes / checkpoints / speed / VRAM

- **One model, ~0.86B params total** (HF safetensors: 859,922,360 params, F32 ≈ **3.44 GB** download). It is not split into tiny/small/large the way SAM2 was; the "size" comes from the shared PE backbone + detector + tracker. SAM 3.1 is a drop-in improved checkpoint set.
- **Latency (paper): 30 ms for a single image with 100+ objects — on an NVIDIA H200.** Video latency scales with object count; "near real-time" for **~5 concurrent objects** on that class of GPU.
- Interactive image FPS (SA-37 bench, big GPU): SAM3 **43.5 FPS** vs SAM2.1-L 93 FPS vs SAM1-H 41 FPS — i.e. SAM3 is roughly half the speed of SAM2 even on datacenter hardware.
- **VRAM:** ~3.4 GB just for F32 weights, plus activations; realistically **8 GB+** GPU for comfortable image inference, more for video/tracking with many objects. These numbers are NVIDIA-class; nothing about SAM3 is "lightweight."

**Implication:** SAM3's strength (open-vocab "name the object") is exactly what you want, but it is a heavy model. The H200 30 ms number does not translate to a MacBook — see §2.

---

## 2. Running SAM3 locally on Apple Silicon (MPS)

**Feasible for image inference, slow, and video/tracking is CUDA-only.**

- The **official repo assumes CUDA everywhere** (Triton kernels, `@torch.autocast(device_type="cuda")`, `.cuda()` calls). Out of the box it fails on a Mac (HF discussion #11: "Cannot run on Apple Silicon (M4) due to Triton").
- **Image inference does work on MPS** via community PRs (#173, **#400** "Add MPS support for image inference on macOS"), or — easier — by using the **🤗 Transformers implementation** (`Sam3Model.from_pretrained("facebook/sam3")`), which avoids the hard CUDA/Triton deps. Recommended path on Mac.
- Required env: `PYTORCH_ENABLE_MPS_FALLBACK=1`. Some ops (`grid_sample`, decoder FFN autocast) need CPU round-trips or `nullcontext()` guards; a few `RuntimeError: unsupported scalarType` / device-mismatch issues are documented.
- **Video / tracking remains CUDA-only** and raises `NotImplementedError` on MPS/CPU (PR #400 explicitly). So SAM3's memory-tracker — the part that would give you temporal stability across camera moves — is **not available on Mac** today.
- **Measured speed:** PR #400 reports MPS image inference **~6.5 s/image on an M2** (vs ~11 s CPU; only 1.7× faster than CPU). That is **0.15 FPS** — three orders of magnitude away from real-time.
- Emerging Apple-native ports (`mlx-sam3`, `sam3.cpp`) aim to fix this but are early/unverified as of writing.

**Verdict:** A Mac can run SAM3 *image* inference for offline/batch work (auto-labeling frames), but **not** as a live per-frame engine, and **not** its video tracker. Real-time SAM3 streaming requires a separate CUDA GPU box.

### Lighter alternatives (if you drop the open-vocab text requirement)

| Model | Params | Speed | Open-vocab text? | CoreML/edge |
|---|---|---|---|---|
| **SAM2.1** | tiny→large | image encoder fast; tracker good | No (PVS only) | community ONNX/CoreML |
| **MobileSAM** | ~9.7M (TinyViT) | ~10–12 ms/img on GPU; runs on CPU | No | yes |
| **EdgeSAM** | 9.6M, 22.1 GFLOPs | **38 FPS on iPhone 14** (first real-time SAM on edge); CoreML `.mlpackage` shipped | No | **yes, ANE-optimized** |
| **FastSAM** | 68M (YOLOv8) | ~40–64 ms/img | No (prompt-free proposals) | partial |

Key point: **none of the lightweight variants do SAM3's "type a word, get all instances."** They're click/box-prompted. So there's a hard tradeoff: *open-vocab concept naming (SAM3, heavy)* vs *real-time on-device (EdgeSAM/MobileSAM, click-prompted)*.

---

## 3. In-browser inference (ONNX Runtime Web / WebGPU / transformers.js)

**Feasible today for SAM2 / MobileSAM, NOT for SAM3.**

- Working stacks for **SAM2/MobileSAM in the browser**:
  - `sam-web` (npm) — WebGPU + CPU fallback, ships **MobileSAM (45 MB)** and **SAM2 (151 MB)**, OPFS model caching, "**encode once, segment many times**", **~50 ms per click decode**. Framework-agnostic. Repo: `github.com/karlorz/sam-web`.
  - `next-sam` (Geronimi73) and `webgpu-sam2` (lucasgelfond) — ONNX Runtime Web + WebGPU, Web Worker to keep UI responsive, encoder→embeddings, decoder→mask per click.
- Reality checks:
  - **transformers.js WebGPU support for SAM2-class models is weak/experimental**; the working demos use **ONNX Runtime Web directly** with `samexporter`-converted ONNX/ORT models.
  - **WebGPU browser support:** Chrome/Edge 113+ yes; Firefox behind a flag; **Safari historically no** (check current status — Safari 26 may have changed this). Your Mac dev should use Chrome.
  - **SAM3 is not exportable to ONNX-web in any practical form today** — it's 0.86B params with a DETR detector + tracker; far too heavy and not converted. In-browser is realistically **SAM2/MobileSAM territory**, which means **click-prompted, not concept-named**.
- The "encode once / decode per click" pattern is the only thing that's interactive-fast in-browser: the expensive image encoder runs once per frozen frame (~hundreds of ms), then each click decodes in ~50 ms.

**Browser vs local Python server:** In-browser is great for *interactive click segmentation of a paused frame* with zero infra. For SAM3's open-vocab concept segmentation you need a **local Python inference server** (FastAPI + WebSocket/HTTP) — and that server wants a CUDA GPU to be real-time.

---

## 4. Reference architecture: stream canvas frames → SAM → masks → overlay

The naive live loop, and where it bites:

```
PlayCanvas WebGL canvas
   │  (1) capture frame
   ▼
canvas.toDataURL / readPixels / OffscreenCanvas + createImageBitmap
   │  (2) downscale to 512–1024px, throttle to N fps
   ▼
WebSocket (binary JPEG/PNG, or raw RGBA) ──► local FastAPI/uvicorn server
                                              │  SAM3 (CUDA) text prompt(s)
                                              ▼  masks + labels + scores (RLE/PNG)
   ◄───────────────────── WebSocket ─────────┘
   │  (3) draw masks as a 2D overlay layer on top of the canvas
   ▼
PlayCanvas (separate 2D overlay canvas / texture)
```

Practical knobs:

- **Frame capture:** Don't use `canvas.toDataURL` per frame (slow base64). Use an **`OffscreenCanvas` + `createImageBitmap`**, or `gl.readPixels` into a sized-down framebuffer, then post a **transferable ArrayBuffer / Blob** over the WebSocket. Init the PlayCanvas/WebGL context with `preserveDrawingBuffer: true` (or capture in the same rAF tick after render) or you'll read black frames.
- **Resolution:** segment at **512–1024 px**, not native. SAM cost is dominated by the image encoder, which scales with pixels. Upscale masks back on the client.
- **Throttle:** cap at **1–5 fps** of *new* SAM runs; interpolate/hold masks on the other frames. Drop frames if the server is busy (latest-frame-wins queue, depth 1).
- **Embedding cache / decoder-only re-run:** this is the classic SAM2 speed trick — run the **image encoder once**, cache the embedding, then re-run only the **lightweight decoder** for new prompts/clicks (~50 ms). **It only helps when the *image* is static** (same frame, new prompt). As the camera moves the embedding is stale, so for a *moving* camera you pay the full encoder every frame — which is the expensive part. SAM3's PCS also isn't a cheap "decoder-only" call the way a SAM2 click is.
- **Protocol:** binary WebSocket frames; send masks back as **RLE or 1-bit PNG per instance + label + score + a frame id** so the client can discard late results.

**The core tension:** "stream a moving camera through SAM every frame" defeats the embedding-cache trick and demands a full heavy forward pass per frame. That's why for a *static scene* you should move the work offline (§5/§6).

---

## 5. Prior art: real-time SAM in video loops & 3D Gaussian Splatting segmentation

Crucial finding: **the mature way to segment a 3DGS scene is to lift 2D masks into the splats *once, offline*, then segmentation is essentially free at render time** (you just filter/recolor Gaussians by their stored ID). These are not per-frame live-inference systems.

3DGS segmentation methods (all do **offline per-scene preprocessing/training**, then real-time *interaction*):

- **SAGA — Segment Any 3D Gaussians** (`jumpat.github.io/SAGA`, arXiv 2312.00860). Distills SAM 2D masks into a **scale-gated affinity feature per Gaussian**; after training, interactive 3D segmentation in **~4 ms**. Needs per-scene training + post-processing.
- **Gaussian Grouping** (ECCV 2024, `github.com/lkeab/gaussian-grouping`, arXiv 2312.00732). Adds a compact **Identity Encoding** to each Gaussian, supervised by **lifted 2D SAM masks** + a zero-shot tracker for cross-view consistency. Enables open-world segment + edit (remove/inpaint/recolor). Offline training per scene.
- **SAGD / SAGS** (`github.com/XuHu0529/SAGS`, arXiv 2401.17857). **Training-free** pipeline: SAM-mask each rendered view, **Gaussian Decomposition** for clean boundaries, then **multi-view voting** to assign per-Gaussian labels. Offline preprocessing, no per-scene network training.
- **Click-Gaussian** (arXiv 2407.11793). Global-Feature-guided Learning for view-consistent interactive segmentation; ~15× faster Gaussian extraction than SAGA.
- **Feature-3DGS**. Distills the SAM decoder into a per-Gaussian feature field (offline), enabling promptable segmentation at render time.

For your assets: you have `world.ply` (the splats) and `mesh_simplified.ply`/`full_mesh.ply` (collision). SAGD/SAGS-style **training-free lifting** maps naturally onto this — render N views, run SAM3 with text prompts, vote labels onto Gaussians.

Real-time 2D SAM in a video/game loop (per-frame, no 3D lift): the only things that hit interactive FPS are **EdgeSAM (38 FPS iPhone)** and **MobileSAM/SAM2 tracker** on a real GPU — all **click/box-prompted, not concept-named**. There is no known system doing **live per-frame SAM3 open-vocab** at real-time FPS on consumer hardware as of May 2026.

---

## 6. Recommended setups for THIS Mac + PlayCanvas project (ranked)

Your scene is **static** ("Moonbase Greenhouse" is a fixed splat world). That single fact changes everything: you do **not** need to run SAM every frame. Exploit it.

### ⭐ Option A (recommended) — Offline lift SAM3 concept masks into the splats; runtime is inference-free

1. **Offline, once per scene:** programmatically orbit a camera around the splat world, render/export **N views** (you already render in PlayCanvas; or render from the `.ply`).
2. Run **SAM3 with text prompts** ("sofa", "teacup", "chair", "pillow") on those views. Do this either on a **rented/owned CUDA GPU** (fast, 30 ms/img) or **slowly on your Mac via 🤗 Transformers + MPS** (seconds/img, but it's a one-time batch — totally fine offline).
3. **Lift to 3D** with a SAGD/SAGS-style **multi-view voting** (training-free) or Gaussian-Grouping-style identity encodings: assign each Gaussian a concept label.
4. **At runtime in PlayCanvas:** segmentation = **filter/recolor Gaussians by stored label**. **Zero inference, true real-time, perfectly view-consistent**, masks never flicker as the camera moves.

- Latency at runtime: **native frame rate** (segmentation is just an attribute lookup).
- Accuracy: **highest** (3D-consistent, no per-frame jitter; SAM3 open-vocab quality).
- Setup complexity: **high** (offline pipeline + lifting code), but it's the correct architecture for a static scene and matches all the published 3DGS work.

### Option B — In-browser click-to-segment on a paused frame (MobileSAM/SAM2, no server)

Use `sam-web` (WebGPU) for **interactive** segmentation: when the user pauses, run the encoder once on the current frame, then **~50 ms per click**. Good for "click the teacup, get its mask." Zero backend, runs entirely in Chrome on the Mac.

- Latency: ~50 ms/click after a few-hundred-ms encode. Real-time *for clicks*, not for a moving camera.
- Accuracy: good geometric masks, but **no concept naming** (no "segment all pillows" by text) and **2D only** (masks don't persist in 3D).
- Setup complexity: **low**. Best quick win / prototype.

### Option C — Live streaming to a local SAM3 server (only worth it with a CUDA GPU)

Implement §4: capture canvas → throttled WebSocket → **FastAPI SAM3 server on an NVIDIA GPU** → masks back → 2D overlay. SAM3's text prompts give you live "segment all chairs" as you move.

- On a **CUDA box (≥8 GB):** ~1–5 fps of fresh SAM3 masks at 512–1024 px is achievable; usable as a live "concept highlight" overlay, but masks are 2D and will jitter frame-to-frame (no 3D consistency).
- On **Mac-only:** **not viable** — image inference is seconds/frame on MPS and the video tracker is CUDA-only. Don't attempt the live loop on Apple Silicon.
- Latency: capture+net+inference; dominated by the encoder. Accuracy: SAM3 quality but temporally unstable. Setup complexity: **medium–high**, plus you need GPU hardware.

### Bottom line

- **Do Option A** for the real product: it's the only path that is genuinely real-time on a Mac *and* gives 3D-consistent, concept-named segments. Run SAM3 offline (Mac MPS batch is fine, or a cheap cloud GPU hour), lift once, ship inference-free runtime.
- **Do Option B first** as a same-day prototype to validate UX (click-to-segment in-browser, no infra).
- **Only do Option C** if you specifically need *live, dynamic, per-frame* concept masks *and* you have a CUDA GPU; skip it on Mac-only.
- If you ever truly need on-device live 2D masks, the realistic engine is **EdgeSAM (CoreML, 38 FPS)** or **MobileSAM** — but accept click/box prompts instead of SAM3's text concepts.

---

## Sources

SAM3:
- https://github.com/facebookresearch/sam3
- https://github.com/facebookresearch/sam3/blob/main/README.md
- https://ai.meta.com/sam3/
- https://arxiv.org/abs/2511.16719 (paper) / https://arxiv.org/html/2511.16719
- https://huggingface.co/facebook/sam3
- https://huggingface.co/facebook/sam3.1
- https://arxiv.org/pdf/2512.06032 (SAM2→SAM3 gap analysis)
- https://arxiv.org/pdf/2512.04585 (SAM3-I instruction following)

Apple Silicon / MPS:
- https://huggingface.co/facebook/sam3/discussions/11
- https://github.com/facebookresearch/sam3/issues/164
- https://github.com/facebookresearch/sam3/pull/173
- https://github.com/facebookresearch/sam3/pull/400
- https://pytorch.org/docs/stable/notes/mps.html
- https://github.com/pytorch/pytorch/issues/84936

Edge / lightweight SAM:
- https://github.com/chongzhou96/EdgeSAM
- https://arxiv.org/html/2312.06660v3 (EdgeSAM)
- https://arxiv.org/html/2306.14289v2 (MobileSAM)

In-browser:
- https://github.com/karlorz/sam-web
- https://npmx.dev/package/sam-web
- https://github.com/geronimi73/next-sam
- https://lucasgelfond.online/software/webgpu-sam2/
- https://github.com/lucasgelfond/webgpu-sam2
- https://github.com/ibaiGorordo/ONNX-SAM2-Segment-Anything
- https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html
- https://medium.com/@geronimo7/in-browser-image-segmentation-with-segment-anything-model-2-c72680170d92

3DGS segmentation:
- https://jumpat.github.io/SAGA/ (arXiv 2312.00860)
- https://github.com/lkeab/gaussian-grouping (arXiv 2312.00732)
- https://github.com/XuHu0529/SAGS (arXiv 2401.17857, SAGD)
- https://arxiv.org/html/2407.11793 (Click-Gaussian)
