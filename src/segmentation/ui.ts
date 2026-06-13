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
  /** Toggle eraser mode (clicks scrub splats out of the selected objects). */
  onToggleEraser: (active: boolean) => void;
  /** Download labels.json for OpenPreserve to consume. */
  onExportLabels: () => void;
  /** Save author-edited species metadata for an object. */
  onUpdateMetadata: (
    id: string,
    fields: { label: string; scientificName: string; description: string },
  ) => void;
}

const CONCEPTS_STORAGE_KEY = "segmentation.concepts.v1";
const ECOSYSTEM_STORAGE_KEY = "segmentation.ecosystem.v1";
const AUTOLABEL_STORAGE_KEY = "segmentation.autolabel.v1";

// Default concept lists per environment type — the "labeling step" presets.
// Cost scales with views × concepts, so keep each list modest.
const ECOSYSTEM_PRESETS: Record<string, string> = {
  reef: "coral, rock, sand, starfish, fish, sponge",
  wetland: "reeds, cattails, grass, shrub, tree, water",
  savanna: "tree, shrub, grass, rock, termite mound",
  forest: "tree, shrub, fern, fallen log, rock",
  generic: "object",
};

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
  private pointBtn: HTMLButtonElement | null = null;
  private eraserBtn: HTMLButtonElement | null = null;
  private autoLabelCheckbox: HTMLInputElement | null = null;
  // Which object's inline metadata editor is open (survives re-renders).
  private editingId: string | null = null;
  private lastObjects: SegmentedObject[] = [];
  private lastSelected: ReadonlySet<string> = new Set();

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

    // Ecosystem preset drives the default concept list; a saved custom list wins.
    const ecosystem = document.getElementById("seg-ecosystem") as HTMLSelectElement | null;
    const savedEco = localStorage.getItem(ECOSYSTEM_STORAGE_KEY) ?? "reef";
    if (ecosystem) {
      ecosystem.value = savedEco;
      ecosystem.addEventListener("change", () => {
        localStorage.setItem(ECOSYSTEM_STORAGE_KEY, ecosystem.value);
        const preset = ECOSYSTEM_PRESETS[ecosystem.value];
        if (preset) {
          this.concepts.value = preset;
          this.persistConcepts();
          this.updateEstimate();
        }
      });
    }
    this.concepts.value =
      localStorage.getItem(CONCEPTS_STORAGE_KEY) ?? ECOSYSTEM_PRESETS[savedEco] ?? ECOSYSTEM_PRESETS.generic!;

    this.autoLabelCheckbox = document.getElementById("seg-autolabel") as HTMLInputElement | null;
    if (this.autoLabelCheckbox) {
      this.autoLabelCheckbox.checked = localStorage.getItem(AUTOLABEL_STORAGE_KEY) !== "0";
      this.autoLabelCheckbox.addEventListener("change", () => {
        localStorage.setItem(AUTOLABEL_STORAGE_KEY, this.autoLabelCheckbox!.checked ? "1" : "0");
      });
    }
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
    // Mode state lives on the button's class so the system can force a mode off when
    // the other one turns on (point mode and eraser are mutually exclusive).
    this.pointBtn = document.getElementById("seg-point-mode") as HTMLButtonElement | null;
    if (this.pointBtn) {
      this.pointBtn.addEventListener("click", () => {
        const active = !this.pointBtn!.classList.contains("seg-point-active");
        this.setPointModeVisual(active);
        callbacks.onTogglePointMode(active);
      });
    }
    // Eraser mode toggle — visuals are driven by the system (setEraserModeVisual) so
    // the button always reflects the actual mode, however it was changed.
    this.eraserBtn = document.getElementById("seg-eraser") as HTMLButtonElement | null;
    if (this.eraserBtn) {
      this.eraserBtn.addEventListener("click", () => {
        const active = !this.eraserBtn!.classList.contains("seg-point-active");
        callbacks.onToggleEraser(active);
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
        if (this.editingId === deleteButton.dataset.id) this.editingId = null;
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
      // Metadata editor: open / save / cancel.
      const editButton = target.closest<HTMLButtonElement>("button[data-action='edit']");
      if (editButton?.dataset.id) {
        this.editingId = this.editingId === editButton.dataset.id ? null : editButton.dataset.id;
        this.renderList(this.lastObjects, this.lastSelected);
        return;
      }
      const saveButton = target.closest<HTMLButtonElement>("button[data-action='meta-save']");
      if (saveButton?.dataset.id) {
        const row = saveButton.closest<HTMLElement>(".seg-item");
        const get = (cls: string) =>
          (row?.querySelector<HTMLInputElement | HTMLTextAreaElement>(`.${cls}`)?.value ?? "").trim();
        callbacks.onUpdateMetadata(saveButton.dataset.id, {
          label: get("seg-meta-name"),
          scientificName: get("seg-meta-sci"),
          description: get("seg-meta-desc"),
        });
        this.editingId = null;
        return;
      }
      const cancelButton = target.closest<HTMLButtonElement>("button[data-action='meta-cancel']");
      if (cancelButton) {
        this.editingId = null;
        this.renderList(this.lastObjects, this.lastSelected);
        return;
      }
      // Clicks inside the editor form shouldn't toggle the row selection.
      if (target.closest(".seg-editor")) return;
      const row = target.closest<HTMLElement>(".seg-item[data-id]");
      if (row?.dataset.id) callbacks.onToggle(row.dataset.id);
    });
  }

  getPromptText(): string {
    return this.prompt.value.trim();
  }

  setPointModeVisual(active: boolean): void {
    if (!this.pointBtn) return;
    this.pointBtn.textContent = active ? "📍 Point Click Mode (ON)" : "📍 Point Click Mode (off)";
    this.pointBtn.classList.toggle("seg-point-active", active);
  }

  setEraserModeVisual(active: boolean): void {
    if (!this.eraserBtn) return;
    this.eraserBtn.textContent = active ? "🧹 Eraser (ON)" : "🧹 Eraser (off)";
    this.eraserBtn.classList.toggle("seg-point-active", active);
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

  isAutoLabelEnabled(): boolean {
    return this.autoLabelCheckbox?.checked ?? false;
  }

  renderList(objects: SegmentedObject[], selectedIds: ReadonlySet<string>): void {
    this.lastObjects = objects;
    this.lastSelected = selectedIds;
    if (objects.length === 0) {
      this.list.innerHTML = `<li class="seg-empty">No objects yet — run a Batch Segment, or type a concept and press G.</li>`;
      return;
    }
    this.list.innerHTML = objects
      .map((object) => {
        const swatch = `rgb(${rgb255(object.color[0])}, ${rgb255(object.color[1])}, ${rgb255(object.color[2])})`;
        const selected = selectedIds.has(object.id) ? " is-selected" : "";
        const sci = object.scientificName
          ? `<span class="seg-count"><em>${escapeHtml(object.scientificName)}</em></span>`
          : "";
        const editor =
          this.editingId === object.id
            ? `
            <div class="seg-editor" style="grid-column:1/-1;display:flex;flex-direction:column;gap:4px;margin-top:6px;width:100%">
              <input class="seg-meta-name" type="text" placeholder="Common name" value="${escapeHtml(object.label)}" />
              <input class="seg-meta-sci" type="text" placeholder="Scientific name" value="${escapeHtml(object.scientificName ?? "")}" />
              <textarea class="seg-meta-desc" rows="2" placeholder="Description / fun fact (shown to players)">${escapeHtml(object.description ?? "")}</textarea>
              <div style="display:flex;gap:6px">
                <button type="button" data-action="meta-save" data-id="${object.id}">Save</button>
                <button type="button" data-action="meta-cancel">Cancel</button>
              </div>
            </div>`
            : "";
        return `
          <li class="seg-item${selected}" data-id="${object.id}" title="Click to toggle highlight in 3D" aria-pressed="${selectedIds.has(object.id)}" style="flex-wrap:wrap">
            <span class="seg-swatch" style="background:${swatch}"></span>
            <span class="seg-meta">
              <span class="seg-label">${escapeHtml(object.label)}</span>
              ${sci}
              <span class="seg-count">${object.splatIndices.length.toLocaleString()} splats · ${object.sourceViews} view${object.sourceViews > 1 ? "s" : ""}</span>
            </span>
            <span class="seg-actions">
              <button type="button" data-action="edit" data-id="${object.id}" title="Edit species info">✎</button>
              <button type="button" data-action="focus" data-id="${object.id}" title="Frame this object">⤢</button>
              <button type="button" data-action="delete" data-id="${object.id}" title="Delete">✕</button>
            </span>
            ${editor}
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
