import * as pc from "playcanvas";
import { SegmentationApi } from "./api";
import { runBatchSweep, SEGMENT_MIN_SCORE, type CameraRig } from "./batch";
import { captureFrame } from "./capture";
import { liftMask } from "./lift";
import { decodeMask } from "./mask";
import { MaskOverlay, type OverlayLayer } from "./overlay";
import { SegmentationRegistry } from "./registry";
import { SplatGrid } from "./splatIndex";
import { SegmentationUi } from "./ui";
import { SelectionViz } from "./viz";
import type { SegmentPrompts, SegmentedObject } from "./types";

const BATCH_MIN_DISTANCE = 2;
const BATCH_MAX_DISTANCE = 150;
// v2: multi-select. Stores a JSON array of selected ids (v1 was a single id string).
const SELECTED_STORAGE_KEY = "segmentation.selected.v2";

export interface SegmentationDeps {
  app: pc.Application;
  canvas: HTMLCanvasElement;
  camera: pc.Entity;
  splatEntity: pc.Entity;
  cellSize: number;
  sceneBounds: { center: [number, number, number]; span: number };
  cameraRig: CameraRig;
  onFocus: (object: SegmentedObject) => void;
}

export class SegmentationSystem {
  private readonly deps: SegmentationDeps;
  private readonly api = new SegmentationApi();
  private readonly overlay: MaskOverlay;
  private readonly viz: SelectionViz;
  private readonly ui: SegmentationUi;
  private grid: SplatGrid | null = null;
  private registry: SegmentationRegistry | null = null;
  private readonly selectedIds = new Set<string>();
  private busy = false;
  private batchRunning = false;
  private batchCancelled = false;
  private batchAbort: AbortController | null = null;
  // The 2D mask overlay is painted in screen space and can't track the scene in 3D, so it
  // only makes sense while the camera is still. We snapshot the camera position when the
  // overlay is drawn and clear it the moment the camera moves — the persistent 3D point
  // highlight (SelectionViz, built from world-space splat indices) then carries the object.
  private overlayAnchor: pc.Vec3 | null = null;

  constructor(deps: SegmentationDeps) {
    this.deps = deps;
    this.overlay = new MaskOverlay(deps.canvas);
    this.viz = new SelectionViz(deps.app, deps.camera);
    this.ui = new SegmentationUi({
      onSegment: (text) => void this.segment(text),
      onToggle: (id) => this.toggleSelection(id),
      onFocus: (id) => this.focus(id),
      onDelete: (id) => this.deleteObject(id),
      onBatchStart: () => void this.runBatch(),
      onBatchCancel: () => this.cancelBatch(),
    });
  }

  init(): void {
    this.buildIndex();
    this.bindKeyboard();
    this.bindOverlayAutoClear();
    void this.reportServerStatus();
  }

  // Drop the screen-space mask overlay as soon as the camera leaves the pose it was
  // captured from; otherwise the flat painted mask slides off the object as you orbit.
  private bindOverlayAutoClear(): void {
    this.deps.app.on("update", () => {
      if (!this.overlayAnchor || this.batchRunning) return;
      const pos = this.deps.camera.getPosition();
      if (pos.distance(this.overlayAnchor) > 1e-3) this.clearOverlay();
    });
  }

  private clearOverlay(): void {
    this.overlay.clear();
    this.overlayAnchor = null;
  }

  private buildIndex(): void {
    try {
      const start = performance.now();
      this.grid = SplatGrid.fromSplatEntity(this.deps.splatEntity, this.deps.cellSize);
      this.registry = new SegmentationRegistry(this.grid.centers);
      this.viz.setCenters(this.grid.centers);
      this.registry.load();
      this.restoreSelection();
      this.refresh();
      console.info("Segmentation index built", {
        ...this.grid.stats,
        totalMs: Math.round(performance.now() - start),
        restored: this.registry.size(),
      });
    } catch (error) {
      console.error("Failed to build segmentation index", error);
      this.ui.setStatus("Splat index unavailable");
    }
  }

  private async reportServerStatus(): Promise<void> {
    const health = await this.api.health();
    if (health) {
      this.ui.setStatus(`Server ready · ${health.model} (${health.device})`);
    } else {
      this.ui.setStatus(`Mock mode · ${this.api.getServerUrl()} unreachable`);
    }
  }

  private bindKeyboard(): void {
    const keyboard = this.deps.app.keyboard;
    if (!keyboard) return;
    keyboard.on(pc.EVENT_KEYDOWN, (event: pc.KeyboardEvent) => {
      if (event.key !== pc.KEY_G) return;
      if (isTextFieldFocused()) return;
      if (this.batchRunning) return;
      event.event?.preventDefault();
      void this.segment(this.ui.getPromptText());
    });
  }

  private async segment(text: string): Promise<void> {
    if (this.busy || this.batchRunning) return;
    if (!this.grid || !this.registry) {
      this.ui.setStatus("Index not ready yet");
      return;
    }
    const concept = text.trim();
    if (!concept) {
      this.ui.setStatus("Enter a concept to segment");
      return;
    }

    this.busy = true;
    this.ui.setBusy(true, "Segmenting…");
    try {
      const frame = captureFrame(this.deps.canvas, this.deps.camera);
      const prompts: SegmentPrompts = { text: [concept] };

      const response = await this.api.segment(frame, prompts);
      if (response.masks.length === 0) {
        this.ui.setBusy(false, "No masks returned");
        return;
      }

      const layers: OverlayLayer[] = [];
      let liftedCount = 0;
      let dropped = 0;
      for (const maskResult of response.masks) {
        // Client-side confidence guard: a low-score match should never spawn an object.
        if (maskResult.score < SEGMENT_MIN_SCORE) {
          dropped += 1;
          continue;
        }
        const mask = await decodeMask(maskResult, response.width, response.height);
        const lifted = liftMask({ grid: this.grid }, frame.pose, mask, null);
        if (!lifted) continue;
        const { object } = this.registry.upsert(mask.label, lifted.indices, mask.score);
        layers.push({ mask, color: object.color });
        liftedCount += 1;
      }

      this.overlay.draw(layers);
      // Anchor the overlay to the current camera pose so the auto-clear can detect motion.
      this.overlayAnchor = layers.length > 0 ? this.deps.camera.getPosition().clone() : null;
      this.refresh();
      this.registry.save();

      const suffix = this.api.isUsingMock() ? " (mock)" : "";
      const droppedNote = dropped > 0 ? ` · ${dropped} below ${Math.round(SEGMENT_MIN_SCORE * 100)}%` : "";
      this.ui.setBusy(false, `Lifted ${liftedCount}/${response.masks.length} mask(s) in ${response.elapsed_ms}ms${droppedNote}${suffix}`);
    } catch (error) {
      console.error("Segmentation failed", error);
      this.ui.setBusy(false, `Error: ${error instanceof Error ? error.message : String(error)}`);
    } finally {
      this.busy = false;
    }
  }

  private async runBatch(): Promise<void> {
    if (this.batchRunning || this.busy) return;
    if (!this.grid || !this.registry) {
      this.ui.setStatus("Index not ready yet");
      return;
    }
    const concepts = this.ui.getConcepts();
    if (concepts.length === 0) {
      this.ui.setStatus("Add at least one concept to batch segment");
      return;
    }

    const rig = this.deps.cameraRig;
    const restoreView = rig.snapshot();
    // Orbit at the SAME zoomed radius the user is currently seeing, so "what I see
    // standing here is what gets segmented" — not a wide room-span radius that
    // pushes furniture out of frame.
    const framingDistance = clamp(restoreView.distance, BATCH_MIN_DISTANCE, BATCH_MAX_DISTANCE);

    this.batchRunning = true;
    this.batchCancelled = false;
    this.batchAbort = new AbortController();
    this.ui.setBatchRunning(true);
    this.ui.setStatus(`Batch sweep · ${concepts.length} concepts × 16 views`);

    try {
      const result = await runBatchSweep({
        api: this.api,
        grid: this.grid,
        registry: this.registry,
        overlay: this.overlay,
        camera: this.deps.camera,
        canvas: this.deps.canvas,
        rig,
        pivot: this.deps.sceneBounds.center,
        concepts,
        framingDistance,
        signal: this.batchAbort.signal,
        isCancelled: () => this.batchCancelled,
        onProgress: (progress) => this.ui.setProgress(progress),
        onViewSegmented: () => this.refresh(),
      });

      this.registry.save();
      const suffix = this.api.isUsingMock() ? " (mock)" : "";
      if (result.completed) {
        this.ui.setStatus(`Batch complete · ${this.registry.size()} objects from ${result.totalViews} views (${result.callsRun} calls)${suffix}`);
      } else {
        this.ui.setStatus(`Batch cancelled at view ${result.viewsRun}/${result.totalViews} · ${this.registry.size()} objects${suffix}`);
      }
    } catch (error) {
      console.error("Batch sweep failed", error);
      this.registry.save();
      this.ui.setStatus(`Batch error: ${error instanceof Error ? error.message : String(error)}`);
    } finally {
      this.batchRunning = false;
      this.batchAbort = null;
      rig.apply(restoreView);
      this.ui.setBatchRunning(false);
      this.overlay.clear();
    }
  }

  private cancelBatch(): void {
    if (!this.batchRunning) return;
    this.batchCancelled = true;
    this.batchAbort?.abort();
    this.ui.setStatus("Cancelling batch…");
  }

  // Clicking a list row TOGGLES the object in/out of the selection: it lights up (or
  // clears) its splats in 3D (persistent highlight from stored indices). Camera movement
  // is decoupled from selection — use the per-row focus control to reframe an object.
  private toggleSelection(id: string): void {
    if (!this.registry?.get(id)) return;
    if (this.selectedIds.has(id)) this.selectedIds.delete(id);
    else this.selectedIds.add(id);
    this.persistSelection();
    this.refresh();
  }

  // Reframe the camera on a single object on demand (the ⤢ focus control). Does NOT
  // change the selection, so framing and highlighting stay independent.
  private focus(id: string): void {
    const object = this.registry?.get(id);
    if (!object) return;
    this.deps.onFocus(object);
  }

  private restoreSelection(): void {
    this.selectedIds.clear();
    const saved = localStorage.getItem(SELECTED_STORAGE_KEY);
    if (!saved) return;
    try {
      const ids = JSON.parse(saved) as unknown;
      if (!Array.isArray(ids)) return;
      // Only keep highlights on reload for objects that still exist; don't move the
      // camera on boot (that would be jarring) — the focus control reframes on demand.
      for (const id of ids) {
        if (typeof id === "string" && this.registry?.get(id)) this.selectedIds.add(id);
      }
    } catch (error) {
      console.warn("Failed to restore selection", error);
    }
  }

  private persistSelection(): void {
    try {
      if (this.selectedIds.size > 0) {
        localStorage.setItem(SELECTED_STORAGE_KEY, JSON.stringify([...this.selectedIds]));
      } else {
        localStorage.removeItem(SELECTED_STORAGE_KEY);
      }
    } catch (error) {
      console.warn("Failed to persist selection", error);
    }
  }

  private deleteObject(id: string): void {
    if (!this.registry) return;
    this.registry.remove(id);
    if (this.selectedIds.delete(id)) this.persistSelection();
    this.refresh();
    this.registry.save();
  }

  private refresh(): void {
    if (!this.registry) return;
    const objects = this.registry.list();
    this.viz.sync(objects, this.selectedIds);
    this.ui.renderList(objects, this.selectedIds);
  }
}

function isTextFieldFocused(): boolean {
  const active = document.activeElement;
  return active instanceof HTMLInputElement || active instanceof HTMLTextAreaElement;
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}
