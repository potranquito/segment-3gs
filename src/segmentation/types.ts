export type Vec3 = [number, number, number];
export type Box2 = [number, number, number, number];

export interface SegmentPoint {
  x: number;
  y: number;
  label: number;
}

export interface SegmentPrompts {
  text?: string[];
  points?: SegmentPoint[];
  boxes?: Box2[];
}

export interface SegmentRequest {
  image: string;
  width: number;
  height: number;
  prompts: SegmentPrompts;
  multimask: boolean;
}

export interface MaskResult {
  label: string;
  score: number;
  mask_png: string;
  bbox: Box2;
}

export interface SegmentResponse {
  masks: MaskResult[];
  width: number;
  height: number;
  model: string;
  elapsed_ms: number;
}

export interface HealthResponse {
  status: string;
  model: string;
  device: string;
  text_supported: boolean;
}

export interface CameraPose {
  viewMatrix: Float32Array;
  projectionMatrix: Float32Array;
  viewProjMatrix: Float32Array;
  invViewProjMatrix: Float32Array;
  fov: number;
  position: Vec3;
  width: number;
  height: number;
}

export interface CapturedFrame {
  image: string;
  pose: CameraPose;
}

export interface DecodedMask {
  label: string;
  score: number;
  width: number;
  height: number;
  data: Uint8Array;
  bbox: Box2;
}

export interface Aabb {
  min: Vec3;
  max: Vec3;
}

export interface SegmentedObject {
  id: string;
  label: string;
  color: Vec3;
  // Final, pruned membership used for highlight / recolor / focus framing. Derived from
  // the per-Gaussian vote/confidence evidence below (see registry.recomputeMembership).
  splatIndices: Uint32Array;
  // --- Per-Gaussian confidence voting (replaces the old plain index union) ---
  // candidateIndices is the SORTED union of every splat ever selected for this object
  // across all merged views. voteCounts[k] / scoreSums[k] are parallel to
  // candidateIndices[k]: how many views selected that splat, and the accumulated mask
  // score it received (all splats from one mask share that mask's score). splatIndices
  // is the subset of candidateIndices that survives the vote/confidence thresholds.
  candidateIndices: Uint32Array;
  voteCounts: Uint16Array;
  scoreSums: Float32Array;
  centroid: Vec3;
  aabb: Aabb;
  sourceViews: number;
  score: number;
}
