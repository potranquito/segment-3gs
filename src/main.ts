import * as pc from "playcanvas";
import "./style.css";
import { SegmentationSystem } from "./segmentation/system";
import type { CameraRig } from "./segmentation/batch";
import type { SegmentedObject } from "./segmentation/types";

// Swap the scene by replacing public/world.ply (or pointing this at another path).
const SPLAT_URL = "/world.ply";
// Lightweight collision mesh, used ONLY to derive the floor height at startup.
const MESH_URL = "/mesh_simplified.ply";
const CELL_SIZE = 5;
const MIN_DISTANCE = 1.5;
const MAX_DISTANCE = 150;
const PITCH_MIN = -1.35;
const PITCH_MAX = 1.35;
// Standoff floor when framing an object: keeps the camera from burying itself inside a
// small foreground/corner object's splats (which just washes the view out).
const FOCUS_MIN_DISTANCE = 2.5;
const EYE_HEIGHT = 1.6;
const FLOOR_RADIUS = 2.5;
// Zoomed-in starting radius: stand close to the grounded room center looking into the
// small room, not the wide room-span view.
const INITIAL_DISTANCE = 2.0;
// Slight downward tilt: camera sits above the pivot and looks down at an angle.
const INITIAL_PITCH = 0.35;

interface RuntimeState {
  splatEntity: pc.Entity;
  camera: pc.Entity;
  pivot: pc.Vec3;
  yaw: number;
  pitch: number;
  distance: number;
  isPointerDown: boolean;
  batchRunning: boolean;
  // Set by the batch sweep: camera sits at `eye` and only its gaze rotates by (yaw,pitch)
  // looking outward. When null the interactive orbit (lookAt pivot) drives the camera.
  lookAround: { eye: [number, number, number]; yaw: number; pitch: number } | null;
}

async function boot(): Promise<void> {
  const canvas = document.querySelector<HTMLCanvasElement>("#app");
  if (!canvas) throw new Error("Missing #app canvas");

  const app = new pc.Application(canvas, {
    mouse: new pc.Mouse(canvas),
    keyboard: new pc.Keyboard(window),
    touch: new pc.TouchDevice(canvas),
    graphicsDeviceOptions: { preserveDrawingBuffer: true },
  });

  app.setCanvasFillMode(pc.FILLMODE_FILL_WINDOW);
  app.setCanvasResolution(pc.RESOLUTION_AUTO);
  app.scene.ambientLight = new pc.Color(0.45, 0.55, 0.7);
  app.start();

  window.addEventListener("resize", () => app.resizeCanvas());

  const splatEntity = await loadSplat(app, SPLAT_URL);

  const bounds = computeWorldBounds(splatEntity);
  const camera = createCamera(app);

  // Ground the spawn on the floor: the robust center's Y sits mid-room (near the
  // ceiling for orbiting), so we read the floor height from the collision mesh
  // under the scene-center XZ and spawn at standing eye height instead.
  //
  // The mesh is OPTIONAL — a fork that brings only its own `world.ply` (the mesh is
  // gitignored, "bring your own splat") must still boot. If the fetch/parse fails or the
  // file is missing, we fall back to grounding on the splat bounds: the low end of the
  // trimmed Y range is a robust floor proxy, so the spawn still lands at standing eye
  // height instead of mid-room.
  try {
    const meshPositions = await loadMeshPositions(MESH_URL);
    const floorY = computeFloorY(meshPositions, bounds.center.x, bounds.center.z, FLOOR_RADIUS);
    bounds.center.y = floorY + EYE_HEIGHT;
  } catch (error) {
    console.warn("Floor mesh unavailable, grounding on splat bounds instead", error);
    bounds.center.y = bounds.floorY + EYE_HEIGHT;
  }

  const state: RuntimeState = {
    splatEntity,
    camera,
    pivot: bounds.center.clone(),
    yaw: 0,
    pitch: INITIAL_PITCH,
    // Stand close to the scene center, slightly raised, looking down at an angle.
    // User can scroll out / re-aim.
    distance: clamp(INITIAL_DISTANCE, MIN_DISTANCE, MAX_DISTANCE),
    isPointerDown: false,
    batchRunning: false,
    lookAround: null,
  };

  bindCameraInput(app, canvas, state);
  initSegmentation(app, canvas, state, bounds);

  app.on("update", (dt: number) => {
    if (!state.batchRunning) updateCameraPan(app, state, dt);
    updateCamera(state);
  });
}

async function loadSplat(app: pc.Application, url: string): Promise<pc.Entity> {
  const asset = new pc.Asset("scene-splat", "gsplat" as never, { url });
  app.assets.add(asset);

  await new Promise<void>((resolve, reject) => {
    asset.once("load", () => resolve());
    asset.once("error", (error: unknown) => reject(error));
    app.assets.load(asset);
  });

  const entity = new pc.Entity("scene-splat");
  (entity as any).addComponent("gsplat", { asset, unified: false });
  entity.setLocalEulerAngles(0, 0, 180);
  app.root.addChild(entity);
  return entity;
}

// Gaussian-splat scenes contain stray "floater" splats far from the real scene,
// so the raw AABB is inflated and its midpoint drifts off-center. We trim each
// axis to its 2nd-98th percentile (robust to floaters) and take the SPATIAL
// midpoint of that trimmed range. This deliberately tracks the room's spatial
// center rather than the splat-density centroid (which over-weights detail-dense
// surfaces like a bookshelf wall). Percentiles use a strided sample + sort to
// stay cheap (no full 3x2.2M sort).
function computeWorldBounds(entity: pc.Entity): { center: pc.Vec3; span: number; floorY: number } {
  const resource = (entity as unknown as { gsplat?: { resource?: { centers?: Float32Array; gsplatData?: { getCenters(): Float32Array } } } }).gsplat?.resource;
  const raw = resource?.centers ?? resource?.gsplatData?.getCenters();
  if (!raw) throw new Error("Splat resource centers are unavailable");

  const matrix = entity.getWorldTransform();
  const point = new pc.Vec3();
  const count = Math.floor(raw.length / 3);

  // Strided sample of world-space coords, axis-separated, for percentile trimming.
  const target = 250_000;
  const stride = Math.max(1, Math.floor(count / target));
  const sampleCount = Math.ceil(count / stride);
  const sx = new Float32Array(sampleCount);
  const sy = new Float32Array(sampleCount);
  const sz = new Float32Array(sampleCount);
  for (let p = 0, s = 0; p < count; p += stride, s++) {
    const i = p * 3;
    point.set(raw[i]!, raw[i + 1]!, raw[i + 2]!);
    matrix.transformPoint(point, point);
    sx[s] = point.x; sy[s] = point.y; sz[s] = point.z;
  }

  const LO = 0.02;
  const HI = 0.98;
  const trimmedAxis = (values: Float32Array): { lo: number; hi: number } => {
    values.sort();
    const lo = values[Math.floor((values.length - 1) * LO)]!;
    const hi = values[Math.floor((values.length - 1) * HI)]!;
    return { lo, hi };
  };

  const bx = trimmedAxis(sx);
  const by = trimmedAxis(sy);
  const bz = trimmedAxis(sz);

  const center = new pc.Vec3((bx.lo + bx.hi) * 0.5, (by.lo + by.hi) * 0.5, (bz.lo + bz.hi) * 0.5);
  const span = Math.max(bx.hi - bx.lo, by.hi - by.lo, bz.hi - bz.lo);
  // Floor proxy for the no-mesh fallback: the low end of the trimmed Y range. The splat
  // entity's Z=180° flip puts "up" at larger world Y, so the trimmed minimum is the floor.
  const floorY = Number.isFinite(by.lo) ? by.lo : center.y;
  return { center, span: Number.isFinite(span) && span > 0 ? span : MAX_DISTANCE, floorY };
}

const PLY_TYPE_SIZE: Record<string, number> = {
  char: 1, int8: 1, uchar: 1, uint8: 1,
  short: 2, int16: 2, ushort: 2, uint16: 2,
  int: 4, int32: 4, uint: 4, uint32: 4, float: 4, float32: 4,
  double: 8, float64: 8,
};

// Minimal positions-only reader for binary_little_endian PLY. Reads the vertex
// element's x/y/z (any scalar float type) and applies the splat's world flip
// (x,y,z) -> (-x,-y,z) so the mesh lands in the same world space as the splat.
// Faces and other vertex props are skipped via stride. Returns world XYZ triples.
async function loadMeshPositions(url: string): Promise<Float32Array> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Failed to fetch mesh: ${response.status}`);
  const buffer = await response.arrayBuffer();
  const bytes = new Uint8Array(buffer);

  const marker = "end_header\n";
  let headerEnd = -1;
  for (let i = 0; i <= bytes.length - marker.length; i++) {
    let match = true;
    for (let j = 0; j < marker.length; j++) {
      if (bytes[i + j] !== marker.charCodeAt(j)) { match = false; break; }
    }
    if (match) { headerEnd = i + marker.length; break; }
  }
  if (headerEnd < 0) throw new Error("PLY end_header not found");

  const header = new TextDecoder("ascii").decode(bytes.subarray(0, headerEnd));
  const lines = header.split("\n");
  if (!lines.some((l) => l.startsWith("format binary_little_endian"))) {
    throw new Error("Only binary_little_endian PLY is supported");
  }

  let vertexCount = 0;
  let inVertex = false;
  let stride = 0;
  let xOff = -1, yOff = -1, zOff = -1;
  let xType = "", yType = "", zType = "";
  for (const line of lines) {
    const parts = line.trim().split(/\s+/);
    if (parts[0] === "element") {
      inVertex = parts[1] === "vertex";
      if (inVertex) vertexCount = Number(parts[2]);
    } else if (parts[0] === "property" && inVertex) {
      if (parts[1] === "list") throw new Error("List property in vertex element is unsupported");
      const type = parts[1]!;
      const name = parts[2]!;
      const size = PLY_TYPE_SIZE[type];
      if (size === undefined) throw new Error(`Unknown PLY type ${type}`);
      if (name === "x") { xOff = stride; xType = type; }
      else if (name === "y") { yOff = stride; yType = type; }
      else if (name === "z") { zOff = stride; zType = type; }
      stride += size;
    }
  }
  if (xOff < 0 || yOff < 0 || zOff < 0) throw new Error("PLY vertex missing x/y/z");

  const view = new DataView(buffer, headerEnd);
  const readAt = (type: string, off: number): number => {
    switch (type) {
      case "double": case "float64": return view.getFloat64(off, true);
      case "float": case "float32": return view.getFloat32(off, true);
      case "int": case "int32": return view.getInt32(off, true);
      case "uint": case "uint32": return view.getUint32(off, true);
      case "short": case "int16": return view.getInt16(off, true);
      case "ushort": case "uint16": return view.getUint16(off, true);
      case "char": case "int8": return view.getInt8(off);
      default: return view.getUint8(off);
    }
  };

  const out = new Float32Array(vertexCount * 3);
  for (let v = 0; v < vertexCount; v++) {
    const base = v * stride;
    // Splat world flip: negate X and Y (matches the splat entity's Z=180 deg rotation).
    out[v * 3] = -readAt(xType, base + xOff);
    out[v * 3 + 1] = -readAt(yType, base + yOff);
    out[v * 3 + 2] = readAt(zType, base + zOff);
  }
  return out;
}

// Floor height under the scene-center XZ: the low (5th) percentile of mesh-vertex
// Y within FLOOR_RADIUS of the center, robust to a stray low vertex.
function computeFloorY(positions: Float32Array, cx: number, cz: number, radius: number): number {
  const r2 = radius * radius;
  const ys: number[] = [];
  for (let i = 0; i < positions.length; i += 3) {
    const dx = positions[i]! - cx;
    const dz = positions[i + 2]! - cz;
    if (dx * dx + dz * dz <= r2) ys.push(positions[i + 1]!);
  }
  if (ys.length === 0) throw new Error("No mesh vertices near scene center for floor estimate");
  ys.sort((a, b) => a - b);
  return ys[Math.floor((ys.length - 1) * 0.05)]!;
}

function createCamera(app: pc.Application): pc.Entity {
  const camera = new pc.Entity("camera");
  camera.addComponent("camera", {
    clearColor: new pc.Color(0.015, 0.024, 0.04),
    farClip: 500,
    fov: 60,
    nearClip: 0.03,
  });
  app.root.addChild(camera);
  return camera;
}

function bindCameraInput(app: pc.Application, canvas: HTMLCanvasElement, state: RuntimeState): void {
  const mouse = app.mouse;
  if (!mouse) return;

  mouse.on(pc.EVENT_MOUSEDOWN, (event: pc.MouseEvent) => {
    if (state.batchRunning) return;
    // pc.Mouse listens on window, so only start an orbit when the press lands on the canvas
    // (otherwise clicking HUD/panel buttons would steal the cursor via pointer lock).
    if (event.element !== canvas) return;
    state.isPointerDown = true;
    canvas.requestPointerLock?.();
  });

  mouse.on(pc.EVENT_MOUSEUP, () => {
    state.isPointerDown = false;
    document.exitPointerLock?.();
  });

  mouse.on(pc.EVENT_MOUSEMOVE, (event: pc.MouseEvent) => {
    if (state.batchRunning) return;
    if (!state.isPointerDown && document.pointerLockElement !== canvas) return;
    state.yaw -= event.dx * 0.006;
    state.pitch = clamp(state.pitch - event.dy * 0.004, PITCH_MIN, PITCH_MAX);
  });

  mouse.on(pc.EVENT_MOUSEWHEEL, (event: pc.MouseEvent) => {
    if (state.batchRunning) return;
    const factor = state.distance * 0.08;
    state.distance = clamp(state.distance + event.wheelDelta * factor, MIN_DISTANCE, MAX_DISTANCE);
  });
}

function updateCameraPan(app: pc.Application, state: RuntimeState, dt: number): void {
  const keyboard = app.keyboard;
  if (!keyboard) return;
  if (isTextFieldFocused()) return;

  const inputX = Number(keyboard.isPressed(pc.KEY_D) || keyboard.isPressed(pc.KEY_RIGHT)) - Number(keyboard.isPressed(pc.KEY_A) || keyboard.isPressed(pc.KEY_LEFT));
  const inputZ = Number(keyboard.isPressed(pc.KEY_W) || keyboard.isPressed(pc.KEY_UP)) - Number(keyboard.isPressed(pc.KEY_S) || keyboard.isPressed(pc.KEY_DOWN));
  const inputY = Number(keyboard.isPressed(pc.KEY_E)) - Number(keyboard.isPressed(pc.KEY_Q));

  const panStep = Math.max(2, state.distance * 0.55) * dt;

  if (inputX !== 0 || inputZ !== 0) {
    const forward = { x: -Math.sin(state.yaw), z: -Math.cos(state.yaw) };
    const right = { x: Math.cos(state.yaw), z: -Math.sin(state.yaw) };
    const length = Math.hypot(inputX, inputZ);
    state.pivot.x += ((right.x * inputX + forward.x * inputZ) / length) * panStep;
    state.pivot.z += ((right.z * inputX + forward.z * inputZ) / length) * panStep;
  }

  if (inputY !== 0) state.pivot.y += inputY * panStep;
}

function initSegmentation(
  app: pc.Application,
  canvas: HTMLCanvasElement,
  state: RuntimeState,
  bounds: { center: pc.Vec3; span: number },
): void {
  const cameraRig: CameraRig = {
    snapshot: () => ({
      yaw: state.yaw,
      pitch: state.pitch,
      distance: state.distance,
      pivot: [state.pivot.x, state.pivot.y, state.pivot.z],
    }),
    apply: (view) => {
      state.yaw = view.yaw;
      state.pitch = view.pitch;
      state.distance = view.distance;
      state.pivot.set(view.pivot[0], view.pivot[1], view.pivot[2]);
      state.lookAround = view.lookAround
        ? {
            eye: [view.lookAround.eye[0], view.lookAround.eye[1], view.lookAround.eye[2]],
            yaw: view.lookAround.yaw,
            pitch: view.lookAround.pitch,
          }
        : null;
    },
    setBatchRunning: (running) => {
      state.batchRunning = running;
    },
    waitForRender: () =>
      new Promise<void>((resolve) => {
        app.once("frameend", () => resolve());
      }),
  };

  const system = new SegmentationSystem({
    app,
    canvas,
    camera: state.camera,
    splatEntity: state.splatEntity,
    cellSize: CELL_SIZE,
    sceneBounds: { center: [bounds.center.x, bounds.center.y, bounds.center.z], span: bounds.span },
    cameraRig,
    onFocus: (object) => focusOnObject(state, object, bounds.center),
  });
  system.init();
}

// Frame the object so it fills a sensible chunk of the view. The orbit camera sits at
// `pivot + dir(yaw,pitch)*distance` looking at the pivot, so:
//  - distance: the standoff that fits a sphere of radius R inside the vertical FOV is
//    R / tan(fov/2); using the AABB half-diagonal (not the largest axis × 2.4) keeps a
//    big splat count from flinging the camera way out.
//  - direction: aim from the object back toward the scene interior so the camera lands
//    in open space looking at the object, instead of keeping the old yaw and burying
//    itself in the wall/furniture behind a foreground (corner) object.
function focusOnObject(state: RuntimeState, object: SegmentedObject, sceneCenter: pc.Vec3): void {
  state.pivot.set(object.centroid[0], object.centroid[1], object.centroid[2]);

  const dx = object.aabb.max[0] - object.aabb.min[0];
  const dy = object.aabb.max[1] - object.aabb.min[1];
  const dz = object.aabb.max[2] - object.aabb.min[2];
  const radius = 0.5 * Math.hypot(dx, dy, dz);
  if (Number.isFinite(radius) && radius > 0) {
    const fovDeg = state.camera.camera?.fov ?? 60;
    const halfFov = (fovDeg * Math.PI) / 180 / 2;
    const framed = (radius / Math.tan(halfFov)) * 1.3;
    state.distance = clamp(framed, Math.max(MIN_DISTANCE, FOCUS_MIN_DISTANCE), MAX_DISTANCE);
  }

  const toInteriorX = sceneCenter.x - object.centroid[0];
  const toInteriorZ = sceneCenter.z - object.centroid[2];
  const horiz = Math.hypot(toInteriorX, toInteriorZ);
  if (horiz > 0.25) {
    state.yaw = Math.atan2(toInteriorX / horiz, toInteriorZ / horiz);
    state.pitch = clamp(0.22, PITCH_MIN, PITCH_MAX);
  }
}

function updateCamera(state: RuntimeState): void {
  const look = state.lookAround;
  if (look) {
    // Pivot in place: camera stays AT the eye point, gaze rotates outward by (yaw,pitch).
    // forward = (sin(yaw)cos(pitch), sin(pitch), cos(yaw)cos(pitch)); negative pitch dips down.
    const cosPitch = Math.cos(look.pitch);
    state.camera.setPosition(look.eye[0], look.eye[1], look.eye[2]);
    state.camera.lookAt(
      look.eye[0] + Math.sin(look.yaw) * cosPitch,
      look.eye[1] + Math.sin(look.pitch),
      look.eye[2] + Math.cos(look.yaw) * cosPitch,
    );
    return;
  }
  const offset = new pc.Vec3(
    Math.sin(state.yaw) * Math.cos(state.pitch) * state.distance,
    Math.sin(state.pitch) * state.distance,
    Math.cos(state.yaw) * Math.cos(state.pitch) * state.distance,
  );
  state.camera.setPosition(state.pivot.x + offset.x, state.pivot.y + offset.y, state.pivot.z + offset.z);
  state.camera.lookAt(state.pivot);
}

function isTextFieldFocused(): boolean {
  const active = document.activeElement;
  return active instanceof HTMLInputElement || active instanceof HTMLTextAreaElement;
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

void boot().catch((error) => {
  console.error(error);
  const hud = document.querySelector("#hud");
  if (hud) hud.innerHTML = `<strong>Failed to load</strong><span>${String(error)}</span>`;
});
