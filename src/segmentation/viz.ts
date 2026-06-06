import * as pc from "playcanvas";
import { GsplatRecolor, type RecolorMode } from "./recolor";
import type { SegmentedObject, Vec3 } from "./types";

// Cap the highlight point cloud so a 300k-splat object still uploads/draws cheaply.
// Subsampling is uniform over the (sorted) index list — enough to read the shape.
const HIGHLIGHT_POINT_CAP = 40_000;
const HIGHLIGHT_POINT_SIZE = 6.0;
// Let the underlying gaussians read through the highlight so it tints the object instead
// of caking over it as a solid blob.
const HIGHLIGHT_ALPHA = 0.55;

const POINT_SHADER = {
  uniqueName: "seg-highlight-points",
  attributes: { aPosition: pc.SEMANTIC_POSITION },
  vertexGLSL: /* glsl */ `
    attribute vec3 aPosition;
    uniform mat4 matrix_model;
    uniform mat4 matrix_viewProjection;
    uniform float uPointSize;
    void main(void) {
      vec4 worldPos = matrix_model * vec4(aPosition, 1.0);
      gl_Position = matrix_viewProjection * worldPos;
      gl_PointSize = uPointSize;
    }
  `,
  fragmentGLSL: /* glsl */ `
    precision highp float;
    uniform vec3 uColor;
    uniform float uAlpha;
    void main(void) {
      vec2 d = gl_PointCoord - vec2(0.5);
      if (dot(d, d) > 0.25) discard;
      gl_FragColor = vec4(uColor, uAlpha);
    }
  `,
};

interface LabelEntry {
  el: HTMLDivElement;
  centroid: Vec3;
}

export class SelectionViz {
  private readonly app: pc.Application;
  private readonly camera: pc.Entity;
  private readonly root: pc.Entity;
  private readonly layer: pc.Layer;
  // The point-cloud highlight is the SOLE 3D visual for a selected object (no AABB box).
  // Multi-select: one highlight entity per selected id, each in its own object color.
  private readonly highlights = new Map<string, pc.Entity>();
  private centers: Float32Array | null = null;
  // DOM labels anchored to each SELECTED object's 3D centroid, repositioned every frame
  // via the camera projection so they follow the object as the camera orbits/moves.
  private readonly labelRoot: HTMLElement;
  private readonly labels = new Map<string, LabelEntry>();
  private readonly viewProj = new pc.Mat4();
  // TRUE gsplat recolor/isolate of the underlying Gaussians (opt-in, alongside the point
  // cloud). Created lazily in setCenters() once the splat count is known. The toggle below
  // is the panel's "Isolate / Recolor" control.
  private recolor: GsplatRecolor | null = null;
  // Default to tinting the actual object Gaussians (the "mask"); the floating point cloud
  // is opt-in via the "Point highlight" option.
  private recolorMode: RecolorMode = "recolor";
  // Last selection pushed through sync(), so a Gaussian-view mode change can re-evaluate
  // whether the point cloud should be shown without waiting for the next registry refresh.
  private lastObjects: SegmentedObject[] = [];
  private lastSelected: ReadonlySet<string> = new Set();

  constructor(app: pc.Application, camera: pc.Entity) {
    this.app = app;
    this.camera = camera;
    this.root = new pc.Entity("segmentation-viz");
    app.root.addChild(this.root);
    this.bindRecolorToggle();

    this.labelRoot = document.createElement("div");
    this.labelRoot.id = "segmentation-labels";
    document.body.appendChild(this.labelRoot);
    // Project the (few) selected centroids to screen space once per rendered frame.
    app.on("update", () => this.updateLabels());

    // Gaussian splats render in their own pass and paint over anything in the World
    // layer, so the highlight points were getting fully covered. Render the selection
    // viz in a dedicated layer pushed AFTER everything (incl. the splats) and have the
    // camera draw it last — the depth-test-disabled material then sits on top reliably.
    this.layer = new pc.Layer({
      name: "segmentation-overlay",
      opaqueSortMode: pc.SORTMODE_NONE,
      transparentSortMode: pc.SORTMODE_NONE,
    });
    app.scene.layers.push(this.layer);
    const cameraComponent = camera.camera;
    if (cameraComponent) cameraComponent.layers = [...cameraComponent.layers, this.layer.id];
  }

  // World-space splat centers (already transformed by the grid). Highlight points are
  // built directly from these, so the highlight entity stays untransformed at the origin.
  setCenters(centers: Float32Array): void {
    this.centers = centers;
    // Stand up the gsplat recolor now that the splat count is known. The splat entity is
    // named "scene-splat" in main.ts; its gsplat material may not exist until the asset
    // finishes loading, which GsplatRecolor handles lazily on first apply().
    const splatEntity = this.app.root.findByName("scene-splat");
    if (splatEntity instanceof pc.Entity) {
      this.recolor = new GsplatRecolor(this.app, splatEntity, centers.length / 3);
      this.recolor.setMode(this.recolorMode);
    }
  }

  // The panel's mode <select> (off / recolor / isolate). Wired here so the recolor feature
  // is fully self-contained in the viz layer; missing markup is a no-op.
  private bindRecolorToggle(): void {
    const select = document.querySelector<HTMLSelectElement>("#seg-recolor-mode");
    if (!select) return;
    select.value = this.recolorMode;
    const sync = (): void => {
      const value = select.value;
      this.recolorMode = value === "recolor" || value === "isolate" ? value : "off";
      this.recolor?.setMode(this.recolorMode);
      // Re-run highlight logic: the point cloud only shows in "off" (Point highlight) mode.
      this.sync(this.lastObjects, this.lastSelected);
    };
    select.addEventListener("change", sync);
    sync();
  }

  sync(objects: SegmentedObject[], selectedIds: ReadonlySet<string>): void {
    this.lastObjects = objects;
    this.lastSelected = selectedIds;
    // Recolor/isolate the actual Gaussians for the current selection (no-op when mode=off).
    this.recolor?.apply(objects, selectedIds);

    const byId = new Map(objects.map((object) => [object.id, object]));

    // The point cloud is the highlight ONLY in "Point highlight" (off) mode; recolor/isolate
    // tint the real Gaussians instead, so drop any point clouds in those modes.
    const pointCloudActive = this.recolorMode === "off";

    // Drop highlights that are no longer selected, whose object is gone, or that a
    // Gaussian-view mode now supersedes.
    for (const [id, entity] of this.highlights) {
      if (!pointCloudActive || !selectedIds.has(id) || !byId.has(id)) {
        entity.destroy();
        this.highlights.delete(id);
      }
    }
    this.syncLabels(byId, selectedIds);

    // Add a highlight for every newly selected object (point-highlight mode only).
    if (!this.centers || !pointCloudActive) return;
    for (const id of selectedIds) {
      if (this.highlights.has(id)) continue;
      const object = byId.get(id);
      if (!object) continue;
      const entity = this.buildHighlight(object, this.centers);
      if (entity) {
        this.highlights.set(id, entity);
        this.root.addChild(entity);
      }
    }
  }

  // Keep one label per selected object, in the object color. Refresh text/color/centroid
  // each call so a merged object (centroid moved) re-anchors correctly. Positioning happens
  // per frame in updateLabels().
  // Labels are intentionally disabled — the colored mask/highlight is the only selection
  // cue. Kept as a no-op (clearing any strays) so the per-frame projection loop has nothing
  // to position.
  private syncLabels(_byId: Map<string, SegmentedObject>, _selectedIds: ReadonlySet<string>): void {
    for (const [id, entry] of this.labels) {
      entry.el.remove();
      this.labels.delete(id);
    }
  }

  // Per frame: project each selected centroid with the camera's view-projection and place
  // its label. Hide labels whose centroid is behind the camera or outside the viewport.
  private updateLabels(): void {
    if (this.labels.size === 0) return;
    const cam = this.camera.camera;
    if (!cam) return;
    this.viewProj.mul2(cam.projectionMatrix, cam.viewMatrix);
    const m = this.viewProj.data;
    const canvas = this.app.graphicsDevice.canvas as HTMLCanvasElement;
    const rect = canvas.getBoundingClientRect();

    for (const entry of this.labels.values()) {
      const [x, y, z] = entry.centroid;
      const cx = m[0] * x + m[4] * y + m[8] * z + m[12];
      const cy = m[1] * x + m[5] * y + m[9] * z + m[13];
      const cw = m[3] * x + m[7] * y + m[11] * z + m[15];
      // Behind the camera (or on the near plane) → hide.
      if (cw <= 0) {
        entry.el.style.display = "none";
        continue;
      }
      const ndcX = cx / cw;
      const ndcY = cy / cw;
      if (ndcX < -1 || ndcX > 1 || ndcY < -1 || ndcY > 1) {
        entry.el.style.display = "none";
        continue;
      }
      const sx = rect.left + (ndcX * 0.5 + 0.5) * rect.width;
      const sy = rect.top + (1 - (ndcY * 0.5 + 0.5)) * rect.height;
      entry.el.style.display = "";
      // translate to the screen point, then offset by the tag's own size so it sits
      // centered just above the centroid.
      entry.el.style.transform = `translate(${sx}px, ${sy}px) translate(-50%, -130%)`;
    }
  }

  private buildHighlight(object: SegmentedObject, centers: Float32Array): pc.Entity | null {
    const idx = object.splatIndices;
    if (idx.length === 0) return null;

    const step = Math.max(1, Math.ceil(idx.length / HIGHLIGHT_POINT_CAP));
    const count = Math.ceil(idx.length / step);
    const positions = new Float32Array(count * 3);
    let o = 0;
    for (let k = 0; k < idx.length; k += step) {
      const i = idx[k]!;
      positions[o++] = centers[i * 3]!;
      positions[o++] = centers[i * 3 + 1]!;
      positions[o++] = centers[i * 3 + 2]!;
    }

    const mesh = new pc.Mesh(this.app.graphicsDevice);
    mesh.setPositions(positions);
    mesh.update(pc.PRIMITIVE_POINTS, true);

    const material = new pc.ShaderMaterial(POINT_SHADER);
    material.setParameter("uColor", [object.color[0], object.color[1], object.color[2]]);
    material.setParameter("uPointSize", HIGHLIGHT_POINT_SIZE);
    material.setParameter("uAlpha", HIGHLIGHT_ALPHA);
    material.cull = pc.CULLFACE_NONE;
    // Draw on top of the (transparent) gsplats. Gaussian splats render in the blend pass
    // and don't write a depth the points can test against, so depthTest here would leave
    // the highlight hidden behind the splat haze; always-on-top guarantees the object
    // "lights up".
    material.blendType = pc.BLEND_NORMAL;
    material.depthTest = false;
    material.depthWrite = false;
    material.update();

    const entity = new pc.Entity(`seg-highlight-${object.id}`);
    entity.addComponent("render", { meshInstances: [new pc.MeshInstance(mesh, material)], layers: [this.layer.id] });
    return entity;
  }
}
