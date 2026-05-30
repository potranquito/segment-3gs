import type { CapturedFrame, HealthResponse, SegmentPrompts, SegmentRequest, SegmentResponse } from "./types";

const DEFAULT_SERVER = "http://localhost:8765";
const HEALTH_TIMEOUT_MS = 1500;
const SEGMENT_TIMEOUT_MS = 60000;

export type ApiMode = "auto" | "real" | "mock";

export interface SegmentOptions {
  multimask?: boolean;
  signal?: AbortSignal;
  timeoutMs?: number;
}

export function resolveServerUrl(): string {
  const fromQuery = new URLSearchParams(location.search).get("server");
  if (fromQuery) return fromQuery.replace(/\/$/, "");
  const fromStorage = localStorage.getItem("segmentation.server");
  if (fromStorage) return fromStorage.replace(/\/$/, "");
  return DEFAULT_SERVER;
}

export function resolveMode(): ApiMode {
  const fromQuery = new URLSearchParams(location.search).get("segmentMode");
  if (fromQuery === "real" || fromQuery === "mock" || fromQuery === "auto") return fromQuery;
  return "auto";
}

export class SegmentationApi {
  private readonly serverUrl: string;
  private mode: ApiMode;
  private healthCache: HealthResponse | null = null;
  private serverReachable: boolean | null = null;

  constructor(serverUrl = resolveServerUrl(), mode: ApiMode = resolveMode()) {
    this.serverUrl = serverUrl;
    this.mode = mode;
  }

  getServerUrl(): string {
    return this.serverUrl;
  }

  getMode(): ApiMode {
    return this.mode;
  }

  isUsingMock(): boolean {
    return this.mode === "mock" || (this.mode === "auto" && this.serverReachable === false);
  }

  async health(): Promise<HealthResponse | null> {
    if (this.mode === "mock") return null;
    try {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), HEALTH_TIMEOUT_MS);
      const response = await fetch(`${this.serverUrl}/health`, { signal: controller.signal });
      clearTimeout(timer);
      if (!response.ok) throw new Error(`health ${response.status}`);
      this.healthCache = (await response.json()) as HealthResponse;
      this.serverReachable = true;
      return this.healthCache;
    } catch {
      this.serverReachable = false;
      return null;
    }
  }

  async segment(frame: CapturedFrame, prompts: SegmentPrompts, options: SegmentOptions = {}): Promise<SegmentResponse> {
    const request: SegmentRequest = {
      image: frame.image,
      width: frame.pose.width,
      height: frame.pose.height,
      prompts,
      multimask: options.multimask ?? false,
    };

    if (this.mode === "mock") return buildMockResponse(request, prompts);

    if (this.mode === "auto" && this.serverReachable === null) await this.health();
    if (this.mode === "auto" && this.serverReachable === false) return buildMockResponse(request, prompts);

    const external = options.signal;
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), options.timeoutMs ?? SEGMENT_TIMEOUT_MS);
    const onExternalAbort = () => controller.abort();
    if (external) {
      if (external.aborted) controller.abort();
      else external.addEventListener("abort", onExternalAbort, { once: true });
    }

    try {
      const response = await fetch(`${this.serverUrl}/segment`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(request),
        signal: controller.signal,
      });
      if (!response.ok) throw new Error(`segment ${response.status}`);
      this.serverReachable = true;
      return (await response.json()) as SegmentResponse;
    } catch (error) {
      // A caller-driven cancel must propagate, never silently degrade to a mock mask.
      if (external?.aborted) throw error;
      if (this.mode === "real") throw error;
      this.serverReachable = false;
      console.warn("Segmentation server unreachable, using mock response", error);
      return buildMockResponse(request, prompts);
    } finally {
      clearTimeout(timer);
      if (external) external.removeEventListener("abort", onExternalAbort);
    }
  }
}

async function buildMockResponse(request: SegmentRequest, prompts: SegmentPrompts): Promise<SegmentResponse> {
  const { width, height } = request;
  const point = prompts.points?.[0];
  const box = prompts.boxes?.[0];
  const label = prompts.text?.[0] ?? (point ? "selection" : "object");

  let cx = width * 0.5;
  let cy = height * 0.5;
  if (point) {
    cx = point.x;
    cy = point.y;
  } else if (box) {
    cx = (box[0] + box[2]) * 0.5;
    cy = (box[1] + box[3]) * 0.5;
  }

  const rx = Math.max(24, Math.min(width, height) * 0.16);
  const ry = rx * 0.78;
  const mask_png = await renderEllipsePng(width, height, cx, cy, rx, ry);
  const bbox: [number, number, number, number] = [
    Math.max(0, cx - rx),
    Math.max(0, cy - ry),
    Math.min(width, cx + rx),
    Math.min(height, cy + ry),
  ];

  return {
    masks: [{ label, score: 0.85, mask_png, bbox }],
    width,
    height,
    model: "mock-ellipse",
    elapsed_ms: 8,
  };
}

async function renderEllipsePng(width: number, height: number, cx: number, cy: number, rx: number, ry: number): Promise<string> {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Failed to create 2D context for mock mask");
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, width, height);
  ctx.fillStyle = "#fff";
  ctx.beginPath();
  ctx.ellipse(cx, cy, rx, ry, 0, 0, Math.PI * 2);
  ctx.fill();
  const dataUrl = canvas.toDataURL("image/png");
  return dataUrl.slice(dataUrl.indexOf(",") + 1);
}
