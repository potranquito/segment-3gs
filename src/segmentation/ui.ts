import { estimateBatch, TOTAL_VIEWS } from "./batch";
import type { BatchProgress } from "./batch";
import type { SegmentedObject } from "./types";

export interface UiCallbacks {
  onSegment: (text: string) => void;
  onToggle: (id: string) => void;
  onFocus: (id: string) => void;
  onDelete: (id: string) => void;
  onBatchStart: () => void;
  onBatchCancel: () => void;
  /** Toggle point-prompt mode (clicks on canvas send SAM a point + current concept). */
  onTogglePointMode: (active: boolean) => void;
  /** Download labels.json for OpenPreserve to consume. */
  onExportLabels: () => void;
}

const CONCEPTS_STORAGE_KEY = "segmentation.concepts.v1";
// Cost scales with views × concepts, so keep the default list modest.
const DEFAULT_CONCEPTS = "sofa, chair, table, cup, pillow, plant, lamp, vase";

export class SegmentationUi {
  private readonly prompt: HTMLInputElement;
  private readonly runButton: HTMLButtonElement;
  private readonly status: HTMLElement;
  private readonly list: HTMLElement;
  private readonly concepts: HTMLTextAreaElement;
  private readonly batchButton: HTMLButtonElement;
  private readonly cancelButton: HTMLButtonElement;
  private readonly progress: HTMLElement;
  private readonly progressBar: HTMLElement;
  private readonly progressText: HTMLElement;
  private readonly estimate: HTMLElement;

  constructor(callbacks: UiCallbacks) {
    this.prompt = requireElement<HTMLInputElement>("#seg-prompt");
    this.runButton = requireElement<HTMLButtonElement>("#seg-run");
    this.status = requireElement<HTMLElement>("#seg-status");
    this.list = requireElement<HTMLElement>("#seg-list");
    this.concepts = requireElement<HTMLTextAreaElement>("#seg-concepts");
    this.batchButton = requireElement<HTMLButtonElement>("#seg-batch");
    this.cancelButton = requireElement<HTMLButtonElement>("#seg-cancel");
    this.progress = requireElement<HTMLElement>("#seg-progress");
    this.progressBar = requireElement<HTMLElement>("#seg-progress-bar");
    this.progressText = requireElement<HTMLElement>("#seg-progress-text");
    this.estimate = requireElement<HTMLElement>("#seg-estimate");

    this.concepts.value = localStorage.getItem(CONCEPTS_STORAGE_KEY) ?? DEFAULT_CONCEPTS;
    this.updateEstimate();
    this.concepts.addEventListener("input", () => this.updateEstimate());
    this.concepts.addEventListener("change", () => this.persistConcepts());
    this.concepts.addEventListener("blur", () => this.persistConcepts());

    this.runButton.addEventListener("click", () => callbacks.onSegment(this.getPromptText()));
    this.prompt.addEventListener("keydown", (event) => {
      if (event.key === "Enter") callbacks.onSegment(this.getPromptText());
    });
    // Point-click mode toggle — when on, clicks in the 3D canvas fire a SAM point prompt
    // (instead of orbiting the camera). The current "Concept" text becomes the label.
    const pointBtn = document.getElementById("seg-point-mode") as HTMLButtonElement | null;
    if (pointBtn) {
      let active = false;
      pointBtn.addEventListener("click", () => {
        active = !active;
        pointBtn.textContent = active ? "📍 Point Click Mode (ON)" : "📍 Point Click Mode (off)";
        pointBtn.classList.toggle("seg-point-active", active);
        callbacks.onTogglePointMode(active);
      });
    }
    // Export labels button — downloads labels.json for OpenPreserve.
    const exportBtn = document.getElementById("seg-export") as HTMLButtonElement | null;
    if (exportBtn) {
      exportBtn.addEventListener("click", () => callbacks.onExportLabels());
    }
    this.batchButton.addEventListener("click", () => callbacks.onBatchStart());
    this.cancelButton.addEventListener("click", () => callbacks.onBatchCancel());
    this.list.addEventListener("click", (event) => {
      const target = event.target;
      if (!(target instanceof HTMLElement)) return;
      const deleteButton = target.closest<HTMLButtonElement>("button[data-action='delete']");
      if (deleteButton?.dataset.id) {
        callbacks.onDelete(deleteButton.dataset.id);
        return;
      }
      // Focus reframes the camera on a single object WITHOUT changing the selection,
      // so toggling several rows never yanks the camera around.
      const focusButton = target.closest<HTMLButtonElement>("button[data-action='focus']");
      if (focusButton?.dataset.id) {
        callbacks.onFocus(focusButton.dataset.id);
        return;
      }
      const row = target.closest<HTMLElement>(".seg-item[data-id]");
      if (row?.dataset.id) callbacks.onToggle(row.dataset.id);
    });
  }

  getPromptText(): string {
    return this.prompt.value.trim();
  }

  getConcepts(): string[] {
    const seen = new Set<string>();
    const out: string[] = [];
    for (const raw of this.concepts.value.split(/[,\n]/)) {
      const phrase = raw.trim();
      if (!phrase) continue;
      const key = phrase.toLowerCase();
      if (seen.has(key)) continue;
      seen.add(key);
      out.push(phrase);
    }
    return out;
  }

  private persistConcepts(): void {
    try {
      localStorage.setItem(CONCEPTS_STORAGE_KEY, this.concepts.value);
    } catch (error) {
      console.warn("Failed to persist concept list", error);
    }
  }

  private updateEstimate(): void {
    const count = this.getConcepts().length;
    const { calls, sam3Label } = estimateBatch(count);
    this.estimate.textContent =
      count === 0
        ? "Add concepts to estimate cost"
        : `${TOTAL_VIEWS} views × ${count} concepts = ${calls} calls · ${sam3Label} on SAM3 · seconds on mobile_sam`;
  }

  setBusy(busy: boolean, message?: string): void {
    this.runButton.disabled = busy;
    this.runButton.classList.toggle("is-busy", busy);
    if (message) this.setStatus(message);
  }

  setBatchRunning(running: boolean): void {
    this.batchButton.disabled = running;
    this.batchButton.classList.toggle("is-busy", running);
    this.runButton.disabled = running;
    this.concepts.disabled = running;
    this.cancelButton.hidden = !running;
    this.progress.hidden = !running;
    this.estimate.hidden = running;
    if (!running) {
      this.progressBar.style.width = "0%";
      this.progressText.textContent = "";
      this.updateEstimate();
    }
  }

  setProgress(progress: BatchProgress): void {
    const totalCalls = progress.totalViews * progress.totalConcepts;
    const done = (progress.viewIndex - 1) * progress.totalConcepts + progress.conceptIndex;
    const ratio = totalCalls > 0 ? Math.min(1, done / totalCalls) : 0;
    this.progressBar.style.width = `${Math.round(ratio * 100)}%`;
    const conceptPart = progress.concept
      ? ` · '${progress.concept}' (${progress.conceptIndex}/${progress.totalConcepts})`
      : "";
    this.progressText.textContent = `View ${progress.viewIndex}/${progress.totalViews}${conceptPart} · ${progress.objectCount} object${progress.objectCount === 1 ? "" : "s"}`;
  }

  setStatus(text: string): void {
    this.status.textContent = text;
  }

  renderList(objects: SegmentedObject[], selectedIds: ReadonlySet<string>): void {
    if (objects.length === 0) {
      this.list.innerHTML = `<li class="seg-empty">No objects yet — run a Batch Segment, or type a concept and press G.</li>`;
      return;
    }
    this.list.innerHTML = objects
      .map((object) => {
        const swatch = `rgb(${rgb255(object.color[0])}, ${rgb255(object.color[1])}, ${rgb255(object.color[2])})`;
        const selected = selectedIds.has(object.id) ? " is-selected" : "";
        return `
          <li class="seg-item${selected}" data-id="${object.id}" title="Click to toggle highlight in 3D" aria-pressed="${selectedIds.has(object.id)}">
            <span class="seg-swatch" style="background:${swatch}"></span>
            <span class="seg-meta">
              <span class="seg-label">${escapeHtml(object.label)}</span>
              <span class="seg-count">${object.splatIndices.length.toLocaleString()} splats · ${object.sourceViews} view${object.sourceViews > 1 ? "s" : ""}</span>
            </span>
            <span class="seg-actions">
              <button type="button" data-action="focus" data-id="${object.id}" title="Frame this object">⤢</button>
              <button type="button" data-action="delete" data-id="${object.id}" title="Delete">✕</button>
            </span>
          </li>`;
      })
      .join("");
  }
}

function requireElement<T extends Element>(selector: string): T {
  const element = document.querySelector<T>(selector);
  if (!element) throw new Error(`Missing element ${selector}`);
  return element;
}

function rgb255(value: number): number {
  return Math.round(value * 255);
}

function escapeHtml(value: string): string {
  return value.replace(/[&<>"']/g, (char) => {
    switch (char) {
      case "&":
        return "&amp;";
      case "<":
        return "&lt;";
      case ">":
        return "&gt;";
      case '"':
        return "&quot;";
      default:
        return "&#39;";
    }
  });
}
