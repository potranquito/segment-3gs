import { pixelToWorldRay, projectWorldPoint } from "./capture";
import { sampleMask, maskCentroid } from "./mask";
import type { SplatGrid } from "./splatIndex";
import type { Aabb, CameraPose, DecodedMask, Vec3 } from "./types";

export interface LiftDeps {
  grid: SplatGrid;
}

export interface LiftOptions {
  depthBand?: number;
}

export interface LiftResult {
  indices: Uint32Array;
  centroid: Vec3;
  aabb: Aabb;
  // The source mask's confidence, carried through so the registry can attribute it to
  // this view's splats. Every splat lifted from one mask shares this single score
  // (uniform per-mask attribution) — the depth/mask test is binary, so there's no
  // per-splat confidence to weight by, and a stronger mask should lend more evidence to
  // all the splats it selects.
  score: number;
}

const DEFAULT_DEPTH_BAND = 2.5;
const ANCHOR_PIXEL_RADIUS = 6;
const BBOX_PIXEL_PAD = 2;

export function liftMask(
  deps: LiftDeps,
  pose: CameraPose,
  mask: DecodedMask,
  anchorPixel: { x: number; y: number } | null,
  options: LiftOptions = {},
): LiftResult | null {
  const depthBand = options.depthBand ?? DEFAULT_DEPTH_BAND;

  // Restrict candidate splats to the screen-frustum slab the mask covers. A 2D mask
  // only spans a thin cone through the world, so the spatial grid lets us touch a
  // small fraction of ~2.2M splats instead of projecting every one. The world AABB is
  // a conservative superset of every splat whose projection can land inside the mask,
  // so the depth/mask test below yields the same selection as the old full scan — it
  // just visits fewer candidates.
  const region = computeMaskWorldAabb(deps.grid, pose, mask, depthBand);
  let culled: number[] | null = null;
  if (region) {
    const collected: number[] = [];
    deps.grid.queryBox(region.min, region.max, (i) => collected.push(i));
    if (collected.length > 0) culled = collected;
  }

  // Degenerate region (mask off-screen math, NaN unprojection, or empty cells) falls
  // back to a full scan so correctness never regresses.
  return scanCandidates(deps.grid, pose, mask, anchorPixel, depthBand, culled);
}

function scanCandidates(
  grid: SplatGrid,
  pose: CameraPose,
  mask: DecodedMask,
  anchorPixel: { x: number; y: number } | null,
  depthBand: number,
  indexList: number[] | null,
): LiftResult | null {
  const anchor = anchorPixel ?? maskCentroid(mask);
  const centers = grid.centers;

  const candidate: number[] = [];
  const candidateDepth: number[] = [];
  let nearAnchorDepth = Infinity;
  let globalMinDepth = Infinity;
  const radiusSq = ANCHOR_PIXEL_RADIUS * ANCHOR_PIXEL_RADIUS;

  const visit = (i: number): void => {
    const x = centers[i * 3]!;
    const y = centers[i * 3 + 1]!;
    const z = centers[i * 3 + 2]!;
    const projected = projectWorldPoint(pose, x, y, z);
    if (projected.depth <= 0) return;
    if (!sampleMask(mask, projected.px, projected.py)) return;
    candidate.push(i);
    candidateDepth.push(projected.depth);
    if (projected.depth < globalMinDepth) globalMinDepth = projected.depth;
    const dx = projected.px - anchor.x;
    const dy = projected.py - anchor.y;
    if (dx * dx + dy * dy <= radiusSq && projected.depth < nearAnchorDepth) {
      nearAnchorDepth = projected.depth;
    }
  };

  if (indexList) {
    for (let k = 0; k < indexList.length; k += 1) visit(indexList[k]!);
  } else {
    grid.forEach(visit);
  }

  if (candidate.length === 0) {
    // The culled region found nothing; retry once over the whole scene before giving up.
    if (indexList) return scanCandidates(grid, pose, mask, anchorPixel, depthBand, null);
    return null;
  }

  // Foreground depth at the clicked point anchors the depth band; fall back to the
  // nearest masked splat when nothing projects close to the anchor pixel.
  const anchorDepth = Number.isFinite(nearAnchorDepth) ? nearAnchorDepth : globalMinDepth;

  const selected: number[] = [];
  for (let k = 0; k < candidate.length; k += 1) {
    if (Math.abs(candidateDepth[k]! - anchorDepth) <= depthBand) selected.push(candidate[k]!);
  }
  if (selected.length === 0) return null;

  const indices = Uint32Array.from(selected);
  const { centroid, aabb } = computeBounds(centers, indices);
  return { indices, centroid, aabb, score: mask.score };
}

// Build a world-space AABB that encloses the truncated frustum the mask's screen bbox
// carves out of the scene's depth range. Returns null when the math is degenerate.
function computeMaskWorldAabb(
  grid: SplatGrid,
  pose: CameraPose,
  mask: DecodedMask,
  depthBand: number,
): { min: Vec3; max: Vec3 } | null {
  const scene = grid.worldBounds;

  // Scene depth range (clip-w) as seen from this camera. clip-w is affine in world
  // coords, so its range over the box is attained at the corners — but only if every
  // corner is in front of the camera. If any corner is behind, bail to a full scan so
  // we never under-estimate the depth slab and miss masked splats.
  let dNear = Infinity;
  let dFar = -Infinity;
  for (let cx = 0; cx < 2; cx += 1) {
    for (let cy = 0; cy < 2; cy += 1) {
      for (let cz = 0; cz < 2; cz += 1) {
        const x = cx ? scene.max[0] : scene.min[0];
        const y = cy ? scene.max[1] : scene.min[1];
        const z = cz ? scene.max[2] : scene.min[2];
        const d = projectWorldPoint(pose, x, y, z).depth;
        if (d <= 0) return null;
        if (d < dNear) dNear = d;
        if (d > dFar) dFar = d;
      }
    }
  }
  if (!Number.isFinite(dNear) || !Number.isFinite(dFar) || dFar <= 0) return null;
  dNear = Math.max(dNear, 0.01);

  // Padded mask bbox in pixel space, clamped to the frame.
  const x0 = Math.max(0, mask.bbox[0] - BBOX_PIXEL_PAD);
  const y0 = Math.max(0, mask.bbox[1] - BBOX_PIXEL_PAD);
  const x1 = Math.min(pose.width, mask.bbox[2] + BBOX_PIXEL_PAD);
  const y1 = Math.min(pose.height, mask.bbox[3] + BBOX_PIXEL_PAD);
  if (x1 <= x0 || y1 <= y0) return null;

  const forward = pixelToWorldRay(pose, pose.width * 0.5, pose.height * 0.5).dir;

  const min: Vec3 = [Infinity, Infinity, Infinity];
  const max: Vec3 = [-Infinity, -Infinity, -Infinity];
  let expanded = false;
  const corners: Array<[number, number]> = [
    [x0, y0],
    [x1, y0],
    [x1, y1],
    [x0, y1],
  ];
  for (const [px, py] of corners) {
    const { origin, dir } = pixelToWorldRay(pose, px, py);
    // depth (clip-w) ≈ distance-along-ray * cos(angle to forward), so the ray distance
    // that reaches a given view depth is depth / cos(angle).
    const cosA = dir[0] * forward[0] + dir[1] * forward[1] + dir[2] * forward[2];
    if (cosA <= 1e-3) continue;
    for (const d of [dNear, dFar]) {
      const t = d / cosA;
      const wx = origin[0] + dir[0] * t;
      const wy = origin[1] + dir[1] * t;
      const wz = origin[2] + dir[2] * t;
      if (wx < min[0]) min[0] = wx;
      if (wy < min[1]) min[1] = wy;
      if (wz < min[2]) min[2] = wz;
      if (wx > max[0]) max[0] = wx;
      if (wy > max[1]) max[1] = wy;
      if (wz > max[2]) max[2] = wz;
      expanded = true;
    }
  }
  if (!expanded) return null;

  const pad = depthBand + grid.cellSizeValue;
  return {
    min: [min[0] - pad, min[1] - pad, min[2] - pad],
    max: [max[0] + pad, max[1] + pad, max[2] + pad],
  };
}

export function computeBounds(centers: Float32Array, indices: Uint32Array): { centroid: Vec3; aabb: Aabb } {
  let sx = 0;
  let sy = 0;
  let sz = 0;
  const min: Vec3 = [Infinity, Infinity, Infinity];
  const max: Vec3 = [-Infinity, -Infinity, -Infinity];
  for (let k = 0; k < indices.length; k += 1) {
    const i = indices[k]!;
    const x = centers[i * 3]!;
    const y = centers[i * 3 + 1]!;
    const z = centers[i * 3 + 2]!;
    sx += x;
    sy += y;
    sz += z;
    if (x < min[0]) min[0] = x;
    if (y < min[1]) min[1] = y;
    if (z < min[2]) min[2] = z;
    if (x > max[0]) max[0] = x;
    if (y > max[1]) max[1] = y;
    if (z > max[2]) max[2] = z;
  }
  const n = indices.length || 1;
  return { centroid: [sx / n, sy / n, sz / n], aabb: { min, max } };
}
