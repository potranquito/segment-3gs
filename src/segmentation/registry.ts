import { computeBounds } from "./lift";
import type { Aabb, SegmentedObject, Vec3 } from "./types";

// Bumped to v2: payload now persists per-Gaussian vote/confidence evidence, not just a
// flat index union. v1 blobs are ignored (load() guards on the version).
const STORAGE_KEY = "segmentation.objects.v2";
const MERGE_OVERLAP_RATIO = 0.25;

// --- Per-Gaussian confidence voting thresholds ---
// A candidate splat is promoted into the final membership (splatIndices) when it was
// selected by at least VOTE_THRESHOLD distinct views, OR its accumulated mask score
// (summed across the views that selected it) reaches SCORE_THRESHOLD. The OR lets a
// single very-confident streak survive while a single noisy view (vote=1, modest score)
// gets pruned once the object has been seen from more than one view.
//
//   VOTE_THRESHOLD = 2  → "seen from >= 2 views" is the primary signal of a real surface.
//   SCORE_THRESHOLD = 1.6 → ~two solid (>=0.8) detections, or strong repeated evidence.
//
// IMPORTANT: a single-view object (sourceViews < 2, i.e. never merged) is NEVER pruned —
// its membership is the full candidate set, so a fresh segmentation always shows
// everything it lifted. Pruning only kicks in once a second view votes.
const VOTE_THRESHOLD = 2;
const SCORE_THRESHOLD = 1.6;

const PALETTE: Vec3[] = [
  [1.0, 0.42, 0.38],
  [0.32, 0.78, 1.0],
  [0.46, 1.0, 0.56],
  [1.0, 0.83, 0.3],
  [0.82, 0.5, 1.0],
  [1.0, 0.6, 0.2],
  [0.4, 1.0, 0.86],
  [1.0, 0.45, 0.78],
];

export class SegmentationRegistry {
  private readonly objects = new Map<string, SegmentedObject>();
  private readonly centers: Float32Array;
  private colorCursor = 0;
  private idCounter = 0;

  constructor(centers: Float32Array) {
    this.centers = centers;
  }

  list(): SegmentedObject[] {
    return [...this.objects.values()];
  }

  get(id: string): SegmentedObject | undefined {
    return this.objects.get(id);
  }

  size(): number {
    return this.objects.size;
  }

  upsert(label: string, indices: Uint32Array, score: number): { object: SegmentedObject; merged: boolean } {
    const sorted = sortIndices(indices);
    const target = this.findMergeTarget(label, sorted);
    if (target) {
      // Merge this view's splats into the candidate evidence: overlapping splats get a
      // vote + their score accumulated, new splats enter with vote=1. All splats from
      // this one mask share the same `score` (uniform per-mask attribution).
      mergeVotes(target, sorted, score);
      target.sourceViews += 1;
      target.score = Math.max(target.score, score);
      this.recomputeMembership(target);
      return { object: target, merged: true };
    }

    // First view of a brand-new object: every splat starts with one vote and this
    // mask's score. Single-view objects keep their full candidate set (no pruning).
    const candidateIndices = sorted;
    const voteCounts = new Uint16Array(sorted.length).fill(1);
    const scoreSums = new Float32Array(sorted.length).fill(score);
    const object: SegmentedObject = {
      id: `seg-${Date.now().toString(36)}-${this.idCounter++}`,
      label,
      color: this.nextColor(),
      splatIndices: sorted,
      candidateIndices,
      voteCounts,
      scoreSums,
      centroid: [0, 0, 0],
      aabb: { min: [0, 0, 0], max: [0, 0, 0] },
      sourceViews: 1,
      score,
    };
    this.recomputeMembership(object);
    this.objects.set(object.id, object);
    return { object, merged: false };
  }

  // Derive the final membership (splatIndices) from the accumulated vote/confidence
  // evidence, then refresh centroid/aabb from that pruned set so focus framing and the
  // highlight match exactly. Single-view objects (never merged) keep every candidate.
  private recomputeMembership(object: SegmentedObject): void {
    const { candidateIndices, voteCounts, scoreSums } = object;
    let membership: Uint32Array;
    if (object.sourceViews < 2) {
      membership = candidateIndices.slice();
    } else {
      const kept: number[] = [];
      for (let k = 0; k < candidateIndices.length; k += 1) {
        if (voteCounts[k]! >= VOTE_THRESHOLD || scoreSums[k]! >= SCORE_THRESHOLD) {
          kept.push(candidateIndices[k]!);
        }
      }
      // Never prune an object to nothing — if the thresholds reject everything (e.g. two
      // views that barely overlap), fall back to the full candidate set.
      membership = kept.length > 0 ? Uint32Array.from(kept) : candidateIndices.slice();
    }
    object.splatIndices = membership;
    const bounds = computeBounds(this.centers, membership);
    object.centroid = bounds.centroid;
    object.aabb = bounds.aabb;
  }

  remove(id: string): void {
    this.objects.delete(id);
  }

  /**
   * Author-editable species metadata. label = common name; empty strings clear
   * the optional fields.
   */
  updateMetadata(
    id: string,
    fields: { label?: string; scientificName?: string; description?: string },
  ): void {
    const object = this.objects.get(id);
    if (!object) return;
    if (fields.label !== undefined && fields.label.trim()) object.label = fields.label.trim();
    if (fields.scientificName !== undefined) {
      object.scientificName = fields.scientificName.trim() || undefined;
    }
    if (fields.description !== undefined) {
      object.description = fields.description.trim() || undefined;
    }
  }

  /**
   * Deep-copy an object's full state so the eraser can offer undo. The copy is
   * detached from the live object (typed arrays sliced, nested arrays cloned).
   */
  snapshotObject(id: string): SegmentedObject | null {
    const o = this.objects.get(id);
    if (!o) return null;
    return {
      ...o,
      color: [o.color[0], o.color[1], o.color[2]],
      splatIndices: o.splatIndices.slice(),
      candidateIndices: o.candidateIndices.slice(),
      voteCounts: o.voteCounts.slice(),
      scoreSums: o.scoreSums.slice(),
      centroid: [o.centroid[0], o.centroid[1], o.centroid[2]],
      aabb: {
        min: [o.aabb.min[0], o.aabb.min[1], o.aabb.min[2]],
        max: [o.aabb.max[0], o.aabb.max[1], o.aabb.max[2]],
      },
    };
  }

  /** Reinsert a snapshot taken by snapshotObject (eraser undo). */
  restoreObject(snapshot: SegmentedObject): void {
    this.objects.set(snapshot.id, snapshot);
  }

  /**
   * Erase every splat of an object inside a world-space sphere. Removal operates on
   * the candidate EVIDENCE (not just the derived membership) — otherwise erased
   * splats would resurrect on the next recomputeMembership/load. Deletes the object
   * outright if nothing survives. Returns the number of candidates removed.
   */
  eraseSphere(id: string, center: Vec3, radius: number): number {
    const object = this.objects.get(id);
    if (!object) return 0;
    const r2 = radius * radius;
    const c = this.centers;
    const n = object.candidateIndices.length;
    const keep: number[] = [];
    for (let k = 0; k < n; k += 1) {
      const i3 = object.candidateIndices[k]! * 3;
      const dx = c[i3]! - center[0];
      const dy = c[i3 + 1]! - center[1];
      const dz = c[i3 + 2]! - center[2];
      if (dx * dx + dy * dy + dz * dz > r2) keep.push(k);
    }
    const removed = n - keep.length;
    if (removed === 0) return 0;
    if (keep.length === 0) {
      this.objects.delete(id);
      return removed;
    }
    const candidateIndices = new Uint32Array(keep.length);
    const voteCounts = new Uint16Array(keep.length);
    const scoreSums = new Float32Array(keep.length);
    for (let i = 0; i < keep.length; i += 1) {
      const k = keep[i]!;
      candidateIndices[i] = object.candidateIndices[k]!;
      voteCounts[i] = object.voteCounts[k]!;
      scoreSums[i] = object.scoreSums[k]!;
    }
    object.candidateIndices = candidateIndices;
    object.voteCounts = voteCounts;
    object.scoreSums = scoreSums;
    this.recomputeMembership(object);
    return removed;
  }

  clear(): void {
    this.objects.clear();
  }

  selectUnderCursor(origin: Vec3, dir: Vec3): SegmentedObject | null {
    let best: SegmentedObject | null = null;
    let bestT = Infinity;
    for (const object of this.objects.values()) {
      const t = rayAabb(origin, dir, object.aabb);
      if (t !== null && t < bestT) {
        bestT = t;
        best = object;
      }
    }
    return best;
  }

  private findMergeTarget(label: string, sorted: Uint32Array): SegmentedObject | null {
    const normalized = label.trim().toLowerCase();
    for (const object of this.objects.values()) {
      if (object.label.trim().toLowerCase() === normalized) return object;
      if (overlapRatio(object.splatIndices, sorted) > MERGE_OVERLAP_RATIO) return object;
    }
    return null;
  }

  // First 8 objects pull hand-tuned palette colors; past that we walk the hue circle by
  // the golden angle so every subsequent object still lands on a visually distinct color
  // instead of repeating the palette.
  private nextColor(): Vec3 {
    const i = this.colorCursor;
    this.colorCursor += 1;
    if (i < PALETTE.length) {
      const color = PALETTE[i]!;
      return [color[0], color[1], color[2]];
    }
    const hue = ((i - PALETTE.length) * 0.61803398875) % 1;
    return hsvToRgb(hue, 0.7, 1.0);
  }

  /**
   * Export all segmented objects as a JSON manifest that another app
   * (OpenPreserve) can consume. World-space centroid + AABB per label,
   * NOT raw Gaussian indices — so the consumer can use any splat
   * representation (Spark.js .sog, gsplat .ply, etc.).
   */
  exportLabelsJson(sceneId: string): string {
    const payload = {
      schema: "openpreserve-labels/v1",
      scene_id: sceneId,
      exported_at: new Date().toISOString(),
      labels: this.list().map((o) => ({
        id: o.id,
        label: o.label,
        color: o.color,
        centroid: o.centroid,
        aabb: o.aabb,
        splat_count: o.splatIndices.length,
        source_views: o.sourceViews,
        confidence: o.score,
        scientific_name: o.scientificName ?? null,
        description: o.description ?? null,
      })),
    };
    return JSON.stringify(payload, null, 2);
  }

  save(): void {
    try {
      // Persist the raw vote/confidence evidence (candidates + votes + scores), not the
      // derived membership. splatIndices is recomputed on load(), so tweaking the
      // thresholds re-prunes existing objects on the next reload.
      const payload = {
        v: 2,
        objects: this.list().map((object) => ({
          id: object.id,
          label: object.label,
          color: object.color,
          sourceViews: object.sourceViews,
          score: object.score,
          scientificName: object.scientificName,
          description: object.description,
          // base64 of the typed-array bytes — compact and lossless.
          idx: encodeBytes(object.candidateIndices),
          votes: encodeBytes(object.voteCounts),
          scores: encodeBytes(object.scoreSums),
        })),
      };
      localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
    } catch (error) {
      console.warn("Failed to persist segmented objects (storage quota?)", error);
    }
  }

  load(): void {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return;
    try {
      const payload = JSON.parse(raw) as {
        v?: number;
        objects: Array<{
          id: string;
          label: string;
          color: Vec3;
          sourceViews: number;
          score: number;
          scientificName?: string;
          description?: string;
          idx: string;
          votes: string;
          scores: string;
        }>;
      };
      if (payload.v !== 2 || !Array.isArray(payload.objects)) return;
      for (const entry of payload.objects) {
        const candidateIndices = new Uint32Array(decodeBytes(entry.idx).buffer);
        const voteCounts = new Uint16Array(decodeBytes(entry.votes).buffer);
        const scoreSums = new Float32Array(decodeBytes(entry.scores).buffer);
        const object: SegmentedObject = {
          id: entry.id,
          label: entry.label,
          color: entry.color,
          splatIndices: candidateIndices.slice(),
          candidateIndices,
          voteCounts,
          scoreSums,
          centroid: [0, 0, 0],
          aabb: { min: [0, 0, 0], max: [0, 0, 0] },
          sourceViews: entry.sourceViews,
          score: entry.score,
          scientificName: entry.scientificName,
          description: entry.description,
        };
        // Re-derive membership + bounds from the persisted evidence under the current
        // thresholds.
        this.recomputeMembership(object);
        this.objects.set(object.id, object);
        this.colorCursor += 1;
      }
    } catch (error) {
      console.warn("Failed to load persisted segmented objects", error);
    }
  }
}

function hsvToRgb(h: number, s: number, v: number): Vec3 {
  const i = Math.floor(h * 6);
  const f = h * 6 - i;
  const p = v * (1 - s);
  const q = v * (1 - f * s);
  const t = v * (1 - (1 - f) * s);
  switch (i % 6) {
    case 0: return [v, t, p];
    case 1: return [q, v, p];
    case 2: return [p, v, t];
    case 3: return [p, q, v];
    case 4: return [t, p, v];
    default: return [v, p, q];
  }
}

function sortIndices(indices: Uint32Array): Uint32Array {
  const copy = Uint32Array.from(indices);
  copy.sort();
  return copy;
}

// Fold a new view's (sorted) splat selection into an object's candidate evidence.
// Overlapping splats gain a vote and accumulate `score`; splats new to this object are
// inserted with vote=1 and `score`. Rebuilds candidateIndices/voteCounts/scoreSums in
// place on the object (sorted-merge, O(n+m)).
function mergeVotes(object: SegmentedObject, incoming: Uint32Array, score: number): void {
  const a = object.candidateIndices;
  const av = object.voteCounts;
  const as = object.scoreSums;
  const b = incoming;
  const max = a.length + b.length;
  const outIdx = new Uint32Array(max);
  const outVotes = new Uint16Array(max);
  const outScores = new Float32Array(max);
  let i = 0;
  let j = 0;
  let n = 0;
  while (i < a.length && j < b.length) {
    const ai = a[i]!;
    const bj = b[j]!;
    if (ai === bj) {
      outIdx[n] = ai;
      outVotes[n] = av[i]! + 1;
      outScores[n] = as[i]! + score;
      n += 1;
      i += 1;
      j += 1;
    } else if (ai < bj) {
      outIdx[n] = ai;
      outVotes[n] = av[i]!;
      outScores[n] = as[i]!;
      n += 1;
      i += 1;
    } else {
      outIdx[n] = bj;
      outVotes[n] = 1;
      outScores[n] = score;
      n += 1;
      j += 1;
    }
  }
  while (i < a.length) {
    outIdx[n] = a[i]!;
    outVotes[n] = av[i]!;
    outScores[n] = as[i]!;
    n += 1;
    i += 1;
  }
  while (j < b.length) {
    outIdx[n] = b[j]!;
    outVotes[n] = 1;
    outScores[n] = score;
    n += 1;
    j += 1;
  }
  object.candidateIndices = outIdx.slice(0, n);
  object.voteCounts = outVotes.slice(0, n);
  object.scoreSums = outScores.slice(0, n);
}

function overlapRatio(a: Uint32Array, b: Uint32Array): number {
  let i = 0;
  let j = 0;
  let overlap = 0;
  while (i < a.length && j < b.length) {
    const av = a[i]!;
    const bv = b[j]!;
    if (av === bv) {
      overlap += 1;
      i += 1;
      j += 1;
    } else if (av < bv) {
      i += 1;
    } else {
      j += 1;
    }
  }
  const denom = Math.min(a.length, b.length) || 1;
  return overlap / denom;
}

function rayAabb(origin: Vec3, dir: Vec3, aabb: Aabb): number | null {
  let tmin = 0;
  let tmax = Infinity;
  for (let axis = 0; axis < 3; axis += 1) {
    const o = origin[axis]!;
    const d = dir[axis]!;
    const lo = aabb.min[axis]!;
    const hi = aabb.max[axis]!;
    if (Math.abs(d) < 1e-8) {
      if (o < lo || o > hi) return null;
    } else {
      let t1 = (lo - o) / d;
      let t2 = (hi - o) / d;
      if (t1 > t2) [t1, t2] = [t2, t1];
      tmin = Math.max(tmin, t1);
      tmax = Math.min(tmax, t2);
      if (tmin > tmax) return null;
    }
  }
  return tmin;
}

// Base64 of a typed array's raw bytes. Works for any typed array (the caller
// reinterprets the decoded bytes with the matching constructor).
function encodeBytes(array: Uint32Array | Uint16Array | Float32Array): string {
  const bytes = new Uint8Array(array.buffer, array.byteOffset, array.byteLength);
  let binary = "";
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunk));
  }
  return btoa(binary);
}

// Decode base64 into a freshly-allocated, 0-offset Uint8Array so its `.buffer` can be
// safely reinterpreted as Uint32/Uint16/Float32 by the caller.
function decodeBytes(encoded: string): Uint8Array {
  const binary = atob(encoded);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
  return bytes;
}
