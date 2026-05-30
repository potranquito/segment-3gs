import * as pc from "playcanvas";
import type { CameraPose, CapturedFrame, Vec3 } from "./types";

export function captureCameraPose(camera: pc.Entity, width: number, height: number): CameraPose {
  const cam = camera.camera;
  if (!cam) throw new Error("Camera entity is missing its camera component");

  const view = cam.viewMatrix.clone();
  const proj = cam.projectionMatrix.clone();
  const viewProj = new pc.Mat4().mul2(proj, view);
  const invViewProj = viewProj.clone().invert();
  const position = camera.getPosition();

  return {
    viewMatrix: new Float32Array(view.data),
    projectionMatrix: new Float32Array(proj.data),
    viewProjMatrix: new Float32Array(viewProj.data),
    invViewProjMatrix: new Float32Array(invViewProj.data),
    fov: cam.fov,
    position: [position.x, position.y, position.z],
    width,
    height,
  };
}

// Cap on the captured frame's longest edge before encoding. The SAM server's cost scales
// with pixel count, and base64 size scales with bytes, so resampling a ~2880px native
// drawing buffer down to <=1024px slashes both with no meaningful loss of segmentation
// quality (masks are scaled back up at draw/lift time). Tune here.
export const CAPTURE_MAX_DIM = 1024;
// JPEG is ~10-40x smaller than lossless PNG for these frames; PIL decodes it server-side.
export const CAPTURE_JPEG_QUALITY = 0.85;

let cachedOffscreen: OffscreenCanvas | null = null;
let cachedCanvas: HTMLCanvasElement | null = null;

function downscaledSize(width: number, height: number): { w: number; h: number } {
  const longEdge = Math.max(width, height);
  const scale = longEdge > CAPTURE_MAX_DIM ? CAPTURE_MAX_DIM / longEdge : 1;
  // Aspect-preserved, never upscale. Round to whole pixels and guard against 0.
  return {
    w: Math.max(1, Math.round(width * scale)),
    h: Math.max(1, Math.round(height * scale)),
  };
}

function getEncodeCanvas(w: number, h: number): HTMLCanvasElement {
  if (!cachedCanvas) cachedCanvas = document.createElement("canvas");
  if (cachedCanvas.width !== w) cachedCanvas.width = w;
  if (cachedCanvas.height !== h) cachedCanvas.height = h;
  return cachedCanvas;
}

// Resample `source` to (w,h) and return base64 JPEG (no `data:` prefix). Prefers an
// OffscreenCanvas for the off-DOM resample; encoding stays synchronous (callers use
// captureFrame synchronously), and OffscreenCanvas has no sync toDataURL, so the final
// JPEG is encoded via a reused detached <canvas>. When OffscreenCanvas is unavailable the
// same <canvas> handles both resample and encode.
function encodeDownscaledJpeg(source: HTMLCanvasElement, w: number, h: number): string {
  let dataUrl: string | null = null;

  if (typeof OffscreenCanvas !== "undefined") {
    if (!cachedOffscreen) cachedOffscreen = new OffscreenCanvas(w, h);
    cachedOffscreen.width = w;
    cachedOffscreen.height = h;
    const octx = cachedOffscreen.getContext("2d");
    if (octx) {
      octx.clearRect(0, 0, w, h);
      octx.drawImage(source, 0, 0, w, h);
      const enc = getEncodeCanvas(w, h);
      const ectx = enc.getContext("2d");
      if (ectx) {
        ectx.clearRect(0, 0, w, h);
        ectx.drawImage(cachedOffscreen, 0, 0);
        dataUrl = enc.toDataURL("image/jpeg", CAPTURE_JPEG_QUALITY);
      }
    }
  }

  if (dataUrl === null) {
    const c = getEncodeCanvas(w, h);
    const ctx = c.getContext("2d");
    if (!ctx) throw new Error("Failed to acquire 2D context for frame downscale");
    ctx.clearRect(0, 0, w, h);
    ctx.drawImage(source, 0, 0, w, h);
    dataUrl = c.toDataURL("image/jpeg", CAPTURE_JPEG_QUALITY);
  }

  return dataUrl.slice(dataUrl.indexOf(",") + 1);
}

export function captureFrame(canvas: HTMLCanvasElement, camera: pc.Entity): CapturedFrame {
  const { w, h } = downscaledSize(canvas.width, canvas.height);
  // Pose dims MUST equal the encoded (= returned mask) dims: the mask comes back at this
  // downscaled resolution and lift.ts maps mask pixels <-> NDC via pose.width/height.
  // Projection matrices are NDC, hence resolution-independent, so the lift stays exact.
  const pose = captureCameraPose(camera, w, h);
  const image = encodeDownscaledJpeg(canvas, w, h);
  return { image, pose };
}

export function projectWorldPoint(pose: CameraPose, x: number, y: number, z: number): { px: number; py: number; depth: number } {
  const m = pose.viewProjMatrix;
  const cx = m[0] * x + m[4] * y + m[8] * z + m[12];
  const cy = m[1] * x + m[5] * y + m[9] * z + m[13];
  const cw = m[3] * x + m[7] * y + m[11] * z + m[15];
  if (cw <= 0) return { px: -1, py: -1, depth: cw };
  const ndcX = cx / cw;
  const ndcY = cy / cw;
  return {
    px: (ndcX * 0.5 + 0.5) * pose.width,
    py: (1 - (ndcY * 0.5 + 0.5)) * pose.height,
    depth: cw,
  };
}

export function pixelToWorldRay(pose: CameraPose, px: number, py: number): { origin: Vec3; dir: Vec3 } {
  const ndcX = (px / pose.width) * 2 - 1;
  const ndcY = 1 - (py / pose.height) * 2;
  const far = unproject(pose.invViewProjMatrix, ndcX, ndcY, 1);
  const origin = pose.position;
  const dx = far[0] - origin[0];
  const dy = far[1] - origin[1];
  const dz = far[2] - origin[2];
  const len = Math.hypot(dx, dy, dz) || 1;
  return { origin: [origin[0], origin[1], origin[2]], dir: [dx / len, dy / len, dz / len] };
}

function unproject(inv: Float32Array, ndcX: number, ndcY: number, ndcZ: number): Vec3 {
  const x = inv[0] * ndcX + inv[4] * ndcY + inv[8] * ndcZ + inv[12];
  const y = inv[1] * ndcX + inv[5] * ndcY + inv[9] * ndcZ + inv[13];
  const z = inv[2] * ndcX + inv[6] * ndcY + inv[10] * ndcZ + inv[14];
  const w = inv[3] * ndcX + inv[7] * ndcY + inv[11] * ndcZ + inv[15];
  const inW = w !== 0 ? 1 / w : 1;
  return [x * inW, y * inW, z * inW];
}
