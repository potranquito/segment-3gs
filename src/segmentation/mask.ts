import type { DecodedMask, MaskResult } from "./types";

export async function decodeMask(mask: MaskResult, width: number, height: number): Promise<DecodedMask> {
  const image = await loadImage(`data:image/png;base64,${mask.mask_png}`);
  const w = image.naturalWidth || width;
  const h = image.naturalHeight || height;
  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  if (!ctx) throw new Error("Failed to create 2D context for mask decode");
  ctx.drawImage(image, 0, 0);
  const rgba = ctx.getImageData(0, 0, w, h).data;
  const data = new Uint8Array(w * h);
  for (let i = 0; i < data.length; i += 1) {
    data[i] = rgba[i * 4] > 127 ? 255 : 0;
  }
  return { label: mask.label, score: mask.score, width: w, height: h, data, bbox: mask.bbox };
}

export function sampleMask(mask: DecodedMask, px: number, py: number): boolean {
  const x = px | 0;
  const y = py | 0;
  if (x < 0 || y < 0 || x >= mask.width || y >= mask.height) return false;
  return mask.data[y * mask.width + x] > 0;
}

export function maskCentroid(mask: DecodedMask): { x: number; y: number } {
  let sx = 0;
  let sy = 0;
  let count = 0;
  for (let y = 0; y < mask.height; y += 1) {
    const row = y * mask.width;
    for (let x = 0; x < mask.width; x += 1) {
      if (mask.data[row + x] > 0) {
        sx += x;
        sy += y;
        count += 1;
      }
    }
  }
  if (count === 0) {
    return { x: (mask.bbox[0] + mask.bbox[2]) * 0.5, y: (mask.bbox[1] + mask.bbox[3]) * 0.5 };
  }
  return { x: sx / count, y: sy / count };
}

function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("Failed to decode mask PNG"));
    image.src = src;
  });
}
