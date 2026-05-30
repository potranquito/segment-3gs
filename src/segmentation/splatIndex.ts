import * as pc from "playcanvas";
import type { Vec3 } from "./types";

export interface SplatGridStats {
  count: number;
  cellSize: number;
  cells: number;
  buildMs: number;
}

export class SplatGrid {
  readonly centers: Float32Array;
  readonly count: number;
  private readonly cellSize: number;
  private readonly minX: number;
  private readonly minY: number;
  private readonly minZ: number;
  private readonly maxX: number;
  private readonly maxY: number;
  private readonly maxZ: number;
  private readonly dimX: number;
  private readonly dimY: number;
  private readonly dimZ: number;
  private readonly cellStart: Int32Array;
  private readonly order: Uint32Array;
  readonly stats: SplatGridStats;

  private constructor(centers: Float32Array, cellSize: number) {
    const start = performance.now();
    this.centers = centers;
    this.count = centers.length / 3;
    this.cellSize = cellSize;

    let minX = Infinity;
    let minY = Infinity;
    let minZ = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    let maxZ = -Infinity;
    for (let i = 0; i < centers.length; i += 3) {
      const x = centers[i]!;
      const y = centers[i + 1]!;
      const z = centers[i + 2]!;
      if (x < minX) minX = x;
      if (y < minY) minY = y;
      if (z < minZ) minZ = z;
      if (x > maxX) maxX = x;
      if (y > maxY) maxY = y;
      if (z > maxZ) maxZ = z;
    }
    this.minX = minX;
    this.minY = minY;
    this.minZ = minZ;
    this.maxX = maxX;
    this.maxY = maxY;
    this.maxZ = maxZ;
    this.dimX = Math.max(1, Math.ceil((maxX - minX) / cellSize) + 1);
    this.dimY = Math.max(1, Math.ceil((maxY - minY) / cellSize) + 1);
    this.dimZ = Math.max(1, Math.ceil((maxZ - minZ) / cellSize) + 1);

    const cellCount = this.dimX * this.dimY * this.dimZ;
    const counts = new Int32Array(cellCount + 1);
    const cellOf = new Int32Array(this.count);
    for (let i = 0; i < this.count; i += 1) {
      const cell = this.cellIndex(centers[i * 3]!, centers[i * 3 + 1]!, centers[i * 3 + 2]!);
      cellOf[i] = cell;
      counts[cell + 1] += 1;
    }
    for (let c = 0; c < cellCount; c += 1) counts[c + 1]! += counts[c]!;
    this.cellStart = counts;
    this.order = new Uint32Array(this.count);
    const cursor = counts.slice();
    for (let i = 0; i < this.count; i += 1) {
      const cell = cellOf[i]!;
      this.order[cursor[cell]!] = i;
      cursor[cell]! += 1;
    }

    this.stats = {
      count: this.count,
      cellSize,
      cells: cellCount,
      buildMs: performance.now() - start,
    };
  }

  static fromSplatEntity(entity: pc.Entity, cellSize: number): SplatGrid {
    const resource = (entity as unknown as { gsplat?: { resource?: { centers?: Float32Array; gsplatData?: { getCenters(): Float32Array } } } }).gsplat?.resource;
    const raw = resource?.centers ?? resource?.gsplatData?.getCenters();
    if (!raw) throw new Error("Splat resource centers are unavailable");

    const world = new Float32Array(raw.length);
    const matrix = entity.getWorldTransform();
    const point = new pc.Vec3();
    for (let i = 0; i < raw.length; i += 3) {
      point.set(raw[i]!, raw[i + 1]!, raw[i + 2]!);
      matrix.transformPoint(point, point);
      world[i] = point.x;
      world[i + 1] = point.y;
      world[i + 2] = point.z;
    }
    return new SplatGrid(world, cellSize);
  }

  get cellSizeValue(): number {
    return this.cellSize;
  }

  get worldBounds(): { min: Vec3; max: Vec3 } {
    return {
      min: [this.minX, this.minY, this.minZ],
      max: [this.maxX, this.maxY, this.maxZ],
    };
  }

  private cellIndex(x: number, y: number, z: number): number {
    const cx = this.clampCell((x - this.minX) / this.cellSize, this.dimX);
    const cy = this.clampCell((y - this.minY) / this.cellSize, this.dimY);
    const cz = this.clampCell((z - this.minZ) / this.cellSize, this.dimZ);
    return (cz * this.dimY + cy) * this.dimX + cx;
  }

  private clampCell(value: number, dim: number): number {
    const v = Math.floor(value);
    if (v < 0) return 0;
    if (v >= dim) return dim - 1;
    return v;
  }

  queryBox(min: Vec3, max: Vec3, visit: (index: number) => void): void {
    const x0 = this.clampCell((min[0] - this.minX) / this.cellSize, this.dimX);
    const y0 = this.clampCell((min[1] - this.minY) / this.cellSize, this.dimY);
    const z0 = this.clampCell((min[2] - this.minZ) / this.cellSize, this.dimZ);
    const x1 = this.clampCell((max[0] - this.minX) / this.cellSize, this.dimX);
    const y1 = this.clampCell((max[1] - this.minY) / this.cellSize, this.dimY);
    const z1 = this.clampCell((max[2] - this.minZ) / this.cellSize, this.dimZ);

    for (let cz = z0; cz <= z1; cz += 1) {
      for (let cy = y0; cy <= y1; cy += 1) {
        const rowBase = (cz * this.dimY + cy) * this.dimX;
        for (let cx = x0; cx <= x1; cx += 1) {
          const cell = rowBase + cx;
          const end = this.cellStart[cell + 1]!;
          for (let k = this.cellStart[cell]!; k < end; k += 1) {
            visit(this.order[k]!);
          }
        }
      }
    }
  }

  forEach(visit: (index: number) => void): void {
    for (let i = 0; i < this.count; i += 1) visit(i);
  }
}
