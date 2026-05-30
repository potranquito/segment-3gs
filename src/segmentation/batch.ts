import type * as pc from "playcanvas";
import type { SegmentationApi } from "./api";
import { captureFrame } from "./capture";
import { liftMask } from "./lift";
import { decodeMask } from "./mask";
import type { MaskOverlay, OverlayLayer } from "./overlay";
import type { SegmentationRegistry } from "./registry";
import type { SplatGrid } from "./splatIndex";
import type { SegmentPrompts } from "./types";

export interface CameraView {
  yaw: number;
  pitch: number;
  distance: number;
  pivot: [number, number, number];
  // When set, the camera is ANCHORED at `eye` and only its gaze rotates by (yaw,pitch)
  // looking OUTWARD — the "stand in the center and turn around" panoramic sweep. This
  // decouples the batch capture from the interactive orbit's lookAt(pivot) model so the
  // camera pivots in place instead of translating around a radius (which swings it into
  // the walls of a small room). The normal orbit path runs when this is absent.
  lookAround?: { eye: [number, number, number]; yaw: number; pitch: number };
}

/**
 * Lets the batch drive the existing orbit camera state. main.ts owns the runtime
 * state + PlayCanvas app; the batch only sets yaw/pitch/distance/pivot and waits for a
 * real render (capture reads the drawn framebuffer).
 */
export interface CameraRig {
  snapshot(): CameraView;
  apply(view: CameraView): void;
  setBatchRunning(running: boolean): void;
  waitForRender(): Promise<void>;
}

export interface SweepPose {
  yaw: number;
  pitch: number;
  distance: number;
}

// 8 yaw angles × 2 pitch angles = 16 outward gaze directions from the fixed center eye.
export const YAW_STEPS = 8;
// Gaze pitch (look-around forward.y = sin(pitch)), so NEGATIVE tilts the gaze DOWN.
// We pick a slightly-down look (~ -14°) and a near-level look (~ -3°) so the sweep
// frames the surrounding walls, furniture, and floor that sit at/below eye level.
// Both are ≤ 0 — we never stare at the ceiling.
const PITCH_VALUES = [-0.25, -0.05];
export const TOTAL_VIEWS = YAW_STEPS * PITCH_VALUES.length;

// SAM3 runs ONE model forward per text phrase (~13s warm on MPS, ~20s cold), and a single
// /segment call can only carry one concept's worth of work safely under the request
// timeout. So the batch issues one HTTP call per concept per view; this per-call timeout
// covers a cold SAM3 forward. mobile_sam is sub-second so the headroom is harmless.
const BATCH_CALL_TIMEOUT_MS = 90000;

// Warm SAM3 forward ≈ 13s/call on MPS. Used only for the upfront UI estimate.
const SAM3_SECONDS_PER_CALL = 13;

// Client-side minimum confidence: drop any returned mask below this before lifting so a
// weak/false match never creates a 3D object. The SAM3 server already thresholds at 0.5
// internally; this is an extra guard the user can raise here. Applied in BOTH the
// interactive Segment-View path (system.ts) and the batch sweep below.
export const SEGMENT_MIN_SCORE = 0.5;

export function estimateBatch(conceptCount: number): { calls: number; sam3Label: string } {
  const calls = TOTAL_VIEWS * conceptCount;
  const minutes = (calls * SAM3_SECONDS_PER_CALL) / 60;
  let sam3Label: string;
  if (calls === 0) sam3Label = "—";
  else if (minutes < 1) sam3Label = "<1 min";
  else sam3Label = `~${Math.round(minutes)} min`;
  return { calls, sam3Label };
}

export function computeSweepPoses(framingDistance: number): SweepPose[] {
  const poses: SweepPose[] = [];
  for (const pitch of PITCH_VALUES) {
    for (let i = 0; i < YAW_STEPS; i += 1) {
      poses.push({ yaw: (i / YAW_STEPS) * Math.PI * 2, pitch, distance: framingDistance });
    }
  }
  return poses;
}

export interface BatchContext {
  api: SegmentationApi;
  grid: SplatGrid;
  registry: SegmentationRegistry;
  overlay: MaskOverlay;
  camera: pc.Entity;
  canvas: HTMLCanvasElement;
  rig: CameraRig;
  pivot: [number, number, number];
  concepts: string[];
  framingDistance: number;
  signal: AbortSignal;
  isCancelled: () => boolean;
  onProgress: (progress: BatchProgress) => void;
  onViewSegmented: () => void;
}

export interface BatchProgress {
  viewIndex: number;
  totalViews: number;
  conceptIndex: number;
  totalConcepts: number;
  concept: string;
  objectCount: number;
}

export interface BatchResult {
  completed: boolean;
  viewsRun: number;
  totalViews: number;
  callsRun: number;
}

export async function runBatchSweep(ctx: BatchContext): Promise<BatchResult> {
  const poses = computeSweepPoses(ctx.framingDistance);
  const totalConcepts = ctx.concepts.length;
  let callsRun = 0;
  ctx.rig.setBatchRunning(true);
  try {
    for (let v = 0; v < poses.length; v += 1) {
      if (ctx.isCancelled()) return cancelled(v, poses.length, callsRun);

      const pose = poses[v]!;
      // Pivot in place: anchor the camera AT the room center and only rotate its gaze
      // outward by (yaw,pitch). The camera position is identical for all 16 poses.
      ctx.rig.apply({
        ...pose,
        pivot: ctx.pivot,
        lookAround: { eye: ctx.pivot, yaw: pose.yaw, pitch: pose.pitch },
      });
      // Two render frames guarantee the update loop repositioned the camera AND the GPU
      // drew it before we read the framebuffer.
      await ctx.rig.waitForRender();
      await ctx.rig.waitForRender();
      if (ctx.isCancelled()) return cancelled(v, poses.length, callsRun);

      // The captured frame is shared across this view's per-concept calls — same pixels,
      // same pose, so lifts from different concepts stay mutually consistent.
      const frame = captureFrame(ctx.canvas, ctx.camera);
      const layers: OverlayLayer[] = [];

      for (let c = 0; c < totalConcepts; c += 1) {
        if (ctx.isCancelled()) {
          ctx.overlay.draw(layers);
          return cancelled(v, poses.length, callsRun);
        }
        const concept = ctx.concepts[c]!;
        ctx.onProgress({
          viewIndex: v + 1,
          totalViews: poses.length,
          conceptIndex: c + 1,
          totalConcepts,
          concept,
          objectCount: ctx.registry.size(),
        });

        // ONE concept per call: SAM3 does a forward per phrase, so a single-concept
        // request stays under the per-call timeout regardless of list length.
        const prompts: SegmentPrompts = { text: [concept] };
        let response;
        try {
          response = await ctx.api.segment(frame, prompts, {
            signal: ctx.signal,
            timeoutMs: BATCH_CALL_TIMEOUT_MS,
          });
        } catch (error) {
          if (ctx.signal.aborted || ctx.isCancelled()) return cancelled(v, poses.length, callsRun);
          console.warn(`Batch view ${v + 1} concept "${concept}" failed; skipping`, error);
          continue;
        }
        callsRun += 1;

        for (const maskResult of response.masks) {
          if (maskResult.score < SEGMENT_MIN_SCORE) continue;
          const mask = await decodeMask(maskResult, response.width, response.height);
          const lifted = liftMask({ grid: ctx.grid }, frame.pose, mask, null);
          if (!lifted) continue;
          const { object } = ctx.registry.upsert(mask.label, lifted.indices, mask.score);
          layers.push({ mask, color: object.color });
        }

        ctx.overlay.draw(layers);
        ctx.onViewSegmented();
      }

      ctx.onProgress({
        viewIndex: v + 1,
        totalViews: poses.length,
        conceptIndex: totalConcepts,
        totalConcepts,
        concept: "",
        objectCount: ctx.registry.size(),
      });
    }
    return { completed: true, viewsRun: poses.length, totalViews: poses.length, callsRun };
  } finally {
    ctx.rig.setBatchRunning(false);
  }
}

function cancelled(viewIndex: number, totalViews: number, callsRun: number): BatchResult {
  return { completed: false, viewsRun: viewIndex, totalViews, callsRun };
}
