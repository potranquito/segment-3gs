import * as pc from "playcanvas";
import type { SegmentedObject } from "./types";

// TRUE gsplat recolor / isolate (the SAGA / Gaussian-Grouping "endgame"): instead of only
// drawing a point-cloud overlay, this tints / hides the underlying Gaussians themselves
// for the selected objects.
//
// HOW IT WORKS (PlayCanvas 2.19, non-unified gsplat):
//   - PlayCanvas exposes user hooks in the splat vertex shader via the `gsplatModifyVS`
//     shader chunk: modifySplatCenter / modifySplatRotationScale / modifySplatColor. The
//     engine calls `modifySplatColor(modelCenter, inout clr)` for every Gaussian, with the
//     global `splat.index` (the linear *source-gaussian* index) in scope.
//   - `splat.index` is EXACTLY the index space the segmentation registry stores in
//     `splatIndices` (both index `resource.centers` / `gsplatData.getCenters()` in source
//     order; the splat entity's Z=180° rotation only moves positions, never reorders the
//     data). So a per-splat lookup texture indexed by `splat.index` lines up 1:1 with the
//     registry's indices — no coordinate remap needed.
//   - We pack an RGBA8 "selection" texture (rgb = object color, a = selected flag) indexed
//     by `splat.index` and override the chunk to: tint members toward their object color
//     and dim the rest ("recolor"), or hide non-members ("isolate") by driving alpha below
//     the engine's alpha-clip threshold (which discards the Gaussian).
//
// PERF: the only added per-Gaussian cost is one branch (when off) or one extra RGBA8
// texture fetch (when on) inside a shader that already does several fetches per splat. The
// selection texture is only re-uploaded when the selection changes, never per frame. This
// stays comfortably real-time on the ~2.2M-splat scene.

export type RecolorMode = "off" | "recolor" | "isolate";

// Mix strength toward the object color for selected Gaussians.
const TINT_STRENGTH = 0.85;
// Brightness multiplier for unselected Gaussians in "recolor" mode (0 = black, 1 = keep).
const DIM_FACTOR = 0.16;
// Selection texture width; height is derived to cover the splat count. 2048 keeps the
// texture well under device limits even for many millions of splats.
const TEX_WIDTH = 2048;

const MODE_VALUE: Record<RecolorMode, number> = { off: 0, recolor: 1, isolate: 2 };

const MODIFY_GLSL = /* glsl */ `
uniform sampler2D uSegTex;
uniform float uSegTexWidth;
uniform float uSegMode;   // 0 = off, 1 = recolor (+dim rest), 2 = isolate (hide rest)
uniform float uSegTint;
uniform float uSegDim;

void modifySplatCenter(inout vec3 center) {}

void modifySplatRotationScale(vec3 originalCenter, vec3 modifiedCenter, inout vec4 rotation, inout vec3 scale) {}

void modifySplatColor(vec3 center, inout vec4 color) {
    if (uSegMode < 0.5) return;
    int w = int(uSegTexWidth);
    if (w <= 0) return;
    int idx = int(splat.index);
    ivec2 uv = ivec2(idx % w, idx / w);
    vec4 seg = texelFetch(uSegTex, uv, 0);
    bool selected = seg.a > 0.5;
    if (uSegMode > 1.5) {
        // Isolate: members keep (tinted) color, everything else is discarded via alpha clip.
        if (!selected) { color.a = 0.0; return; }
        color.rgb = mix(color.rgb, seg.rgb, uSegTint);
    } else {
        // Recolor: tint members toward their object color, dim the rest so they recede.
        if (selected) color.rgb = mix(color.rgb, seg.rgb, uSegTint);
        else color.rgb *= uSegDim;
    }
}
`;

// WGSL mirror. The app defaults to WebGL2 (no deviceTypes set), so this path is normally
// dormant; provided so the feature also works if the device is ever WebGPU.
const MODIFY_WGSL = /* wgsl */ `
var uSegTex: texture_2d<f32>;
uniform uSegTexWidth: f32;
uniform uSegMode: f32;
uniform uSegTint: f32;
uniform uSegDim: f32;

fn modifySplatCenter(center: ptr<function, vec3f>) {}

fn modifySplatRotationScale(originalCenter: vec3f, modifiedCenter: vec3f, rotation: ptr<function, vec4f>, scale: ptr<function, vec3f>) {}

fn modifySplatColor(center: vec3f, color: ptr<function, vec4f>) {
    if (uniform.uSegMode < 0.5) { return; }
    let w: i32 = i32(uniform.uSegTexWidth);
    if (w <= 0) { return; }
    let idx: i32 = i32(splat.index);
    let uv = vec2i(idx % w, idx / w);
    let seg = textureLoad(uSegTex, uv, 0);
    let selected = seg.a > 0.5;
    if (uniform.uSegMode > 1.5) {
        if (!selected) { (*color).a = 0.0; return; }
        (*color) = vec4f(mix((*color).rgb, seg.rgb, uniform.uSegTint), (*color).a);
    } else {
        if (selected) {
            (*color) = vec4f(mix((*color).rgb, seg.rgb, uniform.uSegTint), (*color).a);
        } else {
            (*color) = vec4f((*color).rgb * uniform.uSegDim, (*color).a);
        }
    }
}
`;

export class GsplatRecolor {
  private readonly app: pc.Application;
  private readonly splatEntity: pc.Entity;
  private readonly splatCount: number;
  private mode: RecolorMode = "off";
  private material: pc.ShaderMaterial | null = null;
  private texture: pc.Texture | null = null;
  private pixels: Uint8Array | null = null;
  private texWidth = 0;
  // Cached so a mode toggle can re-apply against the last-known selection.
  private lastObjects: SegmentedObject[] = [];
  private lastSelected: ReadonlySet<string> = new Set();

  constructor(app: pc.Application, splatEntity: pc.Entity, splatCount: number) {
    this.app = app;
    this.splatEntity = splatEntity;
    this.splatCount = splatCount;
  }

  getMode(): RecolorMode {
    return this.mode;
  }

  setMode(mode: RecolorMode): void {
    if (this.mode === mode) return;
    this.mode = mode;
    this.apply(this.lastObjects, this.lastSelected);
  }

  // Called on every selection refresh. Rebuilds the per-splat texture from the selected
  // objects and pushes the current mode to the material.
  apply(objects: SegmentedObject[], selectedIds: ReadonlySet<string>): void {
    this.lastObjects = objects;
    this.lastSelected = selectedIds;
    if (!this.ensureMaterial()) return;

    const selected = objects.filter((o) => selectedIds.has(o.id));
    this.rebuildTexture(selected);

    // Isolate/recolor with an empty selection would blank or dim the whole scene, which
    // reads as "broken". Fall back to off until something is selected.
    const effective = selected.length === 0 ? 0 : MODE_VALUE[this.mode];
    this.material!.setParameter("uSegMode", effective);
  }

  // Acquire the gsplat material (created once the asset/instance loads) and install the
  // chunk override + static uniforms. Idempotent and safe to call before the splat is
  // ready (returns false until it is).
  private ensureMaterial(): boolean {
    const current = (this.splatEntity as unknown as { gsplat?: { material?: pc.ShaderMaterial | null } }).gsplat?.material ?? null;
    if (!current) return false;
    if (this.material === current) return true;

    this.material = current;
    current.getShaderChunks(pc.SHADERLANGUAGE_GLSL).set("gsplatModifyVS", MODIFY_GLSL);
    current.getShaderChunks(pc.SHADERLANGUAGE_WGSL).set("gsplatModifyVS", MODIFY_WGSL);
    current.setParameter("uSegTint", TINT_STRENGTH);
    current.setParameter("uSegDim", DIM_FACTOR);
    current.setParameter("uSegMode", 0);
    const tex = this.ensureTexture();
    current.setParameter("uSegTex", tex);
    current.setParameter("uSegTexWidth", this.texWidth);
    current.update();
    return true;
  }

  private ensureTexture(): pc.Texture {
    if (this.texture) return this.texture;
    const width = TEX_WIDTH;
    const height = Math.max(1, Math.ceil(this.splatCount / width));
    this.texWidth = width;
    this.pixels = new Uint8Array(width * height * 4);
    this.texture = new pc.Texture(this.app.graphicsDevice, {
      name: "seg-selection-id",
      width,
      height,
      format: pc.PIXELFORMAT_RGBA8,
      mipmaps: false,
      minFilter: pc.FILTER_NEAREST,
      magFilter: pc.FILTER_NEAREST,
      addressU: pc.ADDRESS_CLAMP_TO_EDGE,
      addressV: pc.ADDRESS_CLAMP_TO_EDGE,
      levels: [this.pixels],
    });
    return this.texture;
  }

  // Zero the texture, then stamp each selected object's color (+ selected flag in alpha)
  // at the texel for each of its splat indices. The texel byte offset for splat index i is
  // i*4 because the texture is row-major width=texWidth and the shader samples at
  // (i % texWidth, i / texWidth) — the same linear layout. Overlapping splats: last write
  // wins, which is fine.
  private rebuildTexture(selected: SegmentedObject[]): void {
    if (!this.texture || !this.pixels) return;
    const px = this.pixels;
    px.fill(0);
    const count = this.splatCount;
    for (const object of selected) {
      const r = clamp255(object.color[0]);
      const g = clamp255(object.color[1]);
      const b = clamp255(object.color[2]);
      const idx = object.splatIndices;
      for (let k = 0; k < idx.length; k += 1) {
        const i = idx[k]!;
        if (i >= count) continue;
        const o = i * 4;
        px[o] = r;
        px[o + 1] = g;
        px[o + 2] = b;
        px[o + 3] = 255;
      }
    }
    // Push the mutated CPU buffer to the GPU. lock()/unlock() re-uploads level 0.
    const dst = this.texture.lock() as Uint8Array;
    dst.set(px);
    this.texture.unlock();
  }
}

function clamp255(value: number): number {
  return Math.max(0, Math.min(255, Math.round(value * 255)));
}
