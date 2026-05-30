import type { DecodedMask, Vec3 } from "./types";

export interface OverlayLayer {
  mask: DecodedMask;
  color: Vec3;
}

export class MaskOverlay {
  private readonly canvas: HTMLCanvasElement;
  private readonly ctx: CanvasRenderingContext2D;
  private readonly host: HTMLCanvasElement;
  // Mask-resolution scratch buffer; masks are painted here then scaled up onto the overlay.
  private readonly scratch: HTMLCanvasElement;
  private readonly scratchCtx: CanvasRenderingContext2D;

  constructor(host: HTMLCanvasElement) {
    this.host = host;
    this.canvas = document.createElement("canvas");
    this.canvas.id = "segmentation-overlay";
    const ctx = this.canvas.getContext("2d");
    if (!ctx) throw new Error("Failed to create overlay 2D context");
    this.ctx = ctx;
    this.scratch = document.createElement("canvas");
    const sctx = this.scratch.getContext("2d");
    if (!sctx) throw new Error("Failed to create overlay scratch 2D context");
    this.scratchCtx = sctx;
    host.parentElement?.insertBefore(this.canvas, host.nextSibling);
  }

  clear(): void {
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
  }

  draw(layers: OverlayLayer[]): void {
    if (layers.length === 0) {
      this.clear();
      return;
    }
    // Masks arrive at the (downscaled) capture resolution, NOT the display resolution.
    // Size the overlay buffer to the host's render buffer so it's pixel-aligned with the
    // splat view (the shared #segmentation-overlay CSS stretches both to the viewport),
    // then scale each mask up to fill it.
    const width = this.host.width || layers[0]!.mask.width;
    const height = this.host.height || layers[0]!.mask.height;
    if (this.canvas.width !== width || this.canvas.height !== height) {
      this.canvas.width = width;
      this.canvas.height = height;
    }
    this.clear();

    // Transient colored mask feedback only. Object name labels are rendered as 3D-anchored
    // DOM tags (SelectionViz) that track the object as the camera moves, instead of being
    // painted once into this static 2D overlay.
    for (const layer of layers) {
      this.paintMask(layer);
    }
  }

  private paintMask(layer: OverlayLayer): void {
    const { mask, color } = layer;
    if (this.scratch.width !== mask.width || this.scratch.height !== mask.height) {
      this.scratch.width = mask.width;
      this.scratch.height = mask.height;
    }
    const image = this.scratchCtx.createImageData(mask.width, mask.height);
    const data = image.data;
    const r = Math.round(color[0] * 255);
    const g = Math.round(color[1] * 255);
    const b = Math.round(color[2] * 255);
    for (let i = 0; i < mask.data.length; i += 1) {
      if (mask.data[i] > 0) {
        const o = i * 4;
        data[o] = r;
        data[o + 1] = g;
        data[o + 2] = b;
        data[o + 3] = 110;
      }
    }
    this.scratchCtx.putImageData(image, 0, 0);
    // Nearest-neighbour upscale keeps mask edges crisp and avoids alpha bleed where
    // overlapping layers meet. putImageData ignores transforms, hence the scratch + blit.
    this.ctx.imageSmoothingEnabled = false;
    this.ctx.drawImage(this.scratch, 0, 0, mask.width, mask.height, 0, 0, this.canvas.width, this.canvas.height);
  }
}
