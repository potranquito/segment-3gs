"""Model abstraction for the segmentation server.

Three pluggable backends selected via the ``SAM_MODEL`` env var:

- ``sam3``        : Meta SAM3 via 🤗 Transformers (open-vocab text + box prompts).
                    Weights are GATED on HuggingFace and require license acceptance
                    + ``hf auth login``. Points are mapped to tiny boxes (the
                    Transformers SAM3 image path has no native point prompt).
- ``mobile_sam`` /
  ``sam2`` / ``fallback``
                  : ultralytics MobileSAM/FastSAM. Click/box prompts always; text
                    prompts via FastSAM+CLIP when available. Installs from PyPI
                    without gated weights.
- ``mock``        : deterministic fake ellipse masks, zero ML deps. Used for smoke
                    tests and as a stand-in for the frontend.

All segmenters return a list of ``MaskResult`` at the *requested* output
resolution (``width`` x ``height``).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from PIL import Image

logger = logging.getLogger("segmenter")


# --------------------------------------------------------------------------- #
# Shared data structures
# --------------------------------------------------------------------------- #
@dataclass
class Point:
    x: float
    y: float
    label: int = 1  # 1 = foreground, 0 = background


@dataclass
class Prompts:
    text: list[str] = field(default_factory=list)
    points: list[Point] = field(default_factory=list)
    boxes: list[list[float]] = field(default_factory=list)  # [x0, y0, x1, y1]

    @property
    def empty(self) -> bool:
        return not (self.text or self.points or self.boxes)


@dataclass
class MaskResult:
    """A single instance mask in the *output* coordinate space."""

    label: str
    score: float
    mask: np.ndarray  # bool / uint8 array, shape (height, width)
    bbox: tuple[int, int, int, int]  # x0, y0, x1, y1 (pixel coords)


@dataclass
class SegmentResult:
    masks: list[MaskResult]
    message: Optional[str] = None


# --------------------------------------------------------------------------- #
# Device detection (MPS -> CUDA -> CPU). Torch imported lazily.
# --------------------------------------------------------------------------- #
def detect_device() -> str:
    """Return 'mps' | 'cuda' | 'cpu'. Prefers Apple MPS on this Mac, then CUDA."""
    try:
        import torch
    except Exception:  # torch not installed (mock-only install)
        return "cpu"
    try:
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    try:
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
def mask_to_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Tight pixel bbox (x0, y0, x1, y1) of a boolean mask; zeros if empty."""
    ys, xs = np.where(mask)
    if xs.size == 0 or ys.size == 0:
        return (0, 0, 0, 0)
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def resize_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    """Nearest-neighbour resize a boolean/uint8 mask to (height, width)."""
    if mask.shape == (height, width):
        return mask.astype(bool)
    img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    img = img.resize((width, height), Image.NEAREST)
    return np.asarray(img) > 127


# --------------------------------------------------------------------------- #
# Base class
# --------------------------------------------------------------------------- #
class Segmenter(ABC):
    name: str = "base"
    text_supported: bool = False

    def __init__(self, device: str) -> None:
        self.device = device
        self._loaded = False

    @abstractmethod
    def load(self) -> None:
        """Heavy model load. Called lazily on first request."""

    def ensure_loaded(self) -> None:
        if not self._loaded:
            logger.info("Loading model backend '%s' on device '%s'...", self.name, self.device)
            self.load()
            self._loaded = True
            logger.info("Model backend '%s' ready.", self.name)

    @abstractmethod
    def segment(
        self,
        image: Image.Image,
        prompts: Prompts,
        out_width: int,
        out_height: int,
        multimask: bool = False,
    ) -> SegmentResult:
        ...


# --------------------------------------------------------------------------- #
# Mock backend
# --------------------------------------------------------------------------- #
class MockSegmenter(Segmenter):
    """Deterministic ellipse masks. No ML dependencies."""

    name = "mock"
    text_supported = True

    def load(self) -> None:  # nothing to load
        return

    @staticmethod
    def _ellipse(
        width: int,
        height: int,
        cx: float,
        cy: float,
        rx: float,
        ry: float,
    ) -> np.ndarray:
        yy, xx = np.ogrid[:height, :width]
        return ((xx - cx) / max(rx, 1.0)) ** 2 + ((yy - cy) / max(ry, 1.0)) ** 2 <= 1.0

    def _make(
        self, width: int, height: int, label: str, cx: float, cy: float, score: float
    ) -> MaskResult:
        rx, ry = width * 0.18, height * 0.18
        mask = self._ellipse(width, height, cx, cy, rx, ry)
        return MaskResult(label=label, score=score, mask=mask, bbox=mask_to_bbox(mask))

    def segment(
        self,
        image: Image.Image,
        prompts: Prompts,
        out_width: int,
        out_height: int,
        multimask: bool = False,
    ) -> SegmentResult:
        masks: list[MaskResult] = []

        for i, phrase in enumerate(prompts.text):
            # spread phrase ellipses horizontally so multiple are visible
            cx = out_width * (0.3 + 0.4 * (i / max(len(prompts.text), 1)))
            masks.append(self._make(out_width, out_height, phrase, cx, out_height * 0.5, 0.9))

        for i, p in enumerate(prompts.points):
            if p.label == 0:  # background point -> skip producing a mask
                continue
            masks.append(self._make(out_width, out_height, f"point_{i}", p.x, p.y, 0.88))

        for i, box in enumerate(prompts.boxes):
            x0, y0, x1, y1 = box
            masks.append(
                self._make(
                    out_width,
                    out_height,
                    f"object_{i}",
                    (x0 + x1) / 2,
                    (y0 + y1) / 2,
                    0.85,
                )
            )

        if prompts.empty:
            masks.append(
                self._make(out_width, out_height, "object_0", out_width / 2, out_height / 2, 0.8)
            )

        return SegmentResult(masks=masks)


# --------------------------------------------------------------------------- #
# SAM3 backend (gated weights via HF Transformers)
# --------------------------------------------------------------------------- #
class Sam3Segmenter(Segmenter):
    """Meta SAM3 via 🤗 Transformers. Text + box prompts (points -> tiny boxes)."""

    name = "sam3"
    text_supported = True

    # default detection threshold for text/concept prompts
    SCORE_THRESHOLD = 0.5
    MASK_THRESHOLD = 0.5
    POINT_BOX_HALF = 16  # px half-size of the synthetic box for a click point

    def __init__(self, device: str, model_id: str = "facebook/sam3") -> None:
        super().__init__(device)
        self.model_id = model_id
        self._model = None
        self._processor = None
        self._torch = None

    def load(self) -> None:
        try:
            import torch
            from transformers import Sam3Model, Sam3Processor
        except Exception as exc:  # transformers too old / not installed
            raise RuntimeError(
                "SAM3 backend requires a recent 🤗 Transformers (>=5.9) exposing "
                "`Sam3Model`/`Sam3Processor`. Install it:\n"
                "  pip install -U 'transformers>=5.9' torch pillow\n"
                f"Underlying import error: {exc}"
            ) from exc

        self._torch = torch
        try:
            self._model = Sam3Model.from_pretrained(self.model_id).to(self.device)
            self._model.eval()
            self._processor = Sam3Processor.from_pretrained(self.model_id)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load gated SAM3 weights '{self.model_id}'. SAM3 weights are "
                "GATED on HuggingFace. To enable them:\n"
                "  1. Accept the license at https://huggingface.co/facebook/sam3\n"
                "  2. `hf auth login` (or `huggingface-cli login`) with a token\n"
                "  3. Re-run with SAM_MODEL=sam3\n"
                f"Underlying error: {exc}"
            ) from exc

    # -- internal: run one processor() call and collect instance masks -------- #
    def _run(
        self,
        image: Image.Image,
        out_width: int,
        out_height: int,
        text: Optional[str] = None,
        input_boxes: Optional[list[list[float]]] = None,
        input_boxes_labels: Optional[list[int]] = None,
    ) -> list[tuple[float, np.ndarray]]:
        torch = self._torch
        kwargs: dict = {"images": image, "return_tensors": "pt"}
        if text is not None:
            kwargs["text"] = text
        if input_boxes is not None:
            kwargs["input_boxes"] = [input_boxes]  # [batch, num_boxes, 4]
            kwargs["input_boxes_labels"] = [input_boxes_labels or [1] * len(input_boxes)]

        inputs = self._processor(**kwargs).to(self.device)
        with torch.no_grad():
            outputs = self._model(**inputs)

        results = self._processor.post_process_instance_segmentation(
            outputs,
            threshold=self.SCORE_THRESHOLD,
            mask_threshold=self.MASK_THRESHOLD,
            target_sizes=inputs.get("original_sizes").tolist(),
        )[0]

        out: list[tuple[float, np.ndarray]] = []
        masks = results.get("masks", [])
        scores = results.get("scores", [1.0] * len(masks))
        for mask, score in zip(masks, scores):
            arr = mask.detach().cpu().numpy() if hasattr(mask, "detach") else np.asarray(mask)
            arr = arr.astype(bool)
            out.append((float(score), resize_mask(arr, out_width, out_height)))
        return out

    def segment(
        self,
        image: Image.Image,
        prompts: Prompts,
        out_width: int,
        out_height: int,
        multimask: bool = False,
    ) -> SegmentResult:
        self.ensure_loaded()
        results: list[MaskResult] = []

        for phrase in prompts.text:
            for score, mask in self._run(image, out_width, out_height, text=phrase):
                results.append(
                    MaskResult(label=phrase, score=score, mask=mask, bbox=mask_to_bbox(mask))
                )

        for i, box in enumerate(prompts.boxes):
            for score, mask in self._run(
                image, out_width, out_height, input_boxes=[list(box)], input_boxes_labels=[1]
            ):
                results.append(
                    MaskResult(label=f"object_{i}", score=score, mask=mask, bbox=mask_to_bbox(mask))
                )

        # Transformers SAM3 image path has no native point prompt -> synthesize a
        # small box around each foreground click.
        #
        # SAM3 box prompts return EVERY instance its detection head finds (often
        # near-whole-frame masks when the box gives weak signal). A click is an
        # instance-SELECTION gesture, so: keep only masks that contain the clicked
        # pixel, then return the smallest such mask (the most specific instance).
        h = self.POINT_BOX_HALF
        for i, p in enumerate(prompts.points):
            if p.label == 0:
                continue
            synth = [p.x - h, p.y - h, p.x + h, p.y + h]
            candidates = self._run(
                image, out_width, out_height, input_boxes=[synth], input_boxes_labels=[1]
            )
            px = min(max(int(round(p.x)), 0), out_width - 1)
            py = min(max(int(round(p.y)), 0), out_height - 1)
            containing = [(score, mask) for score, mask in candidates if mask[py, px]]
            if not containing:
                continue
            containing.sort(key=lambda sm: int(sm[1].sum()))
            score, mask = containing[0]
            results.append(
                MaskResult(label=f"point_{i}", score=score, mask=mask, bbox=mask_to_bbox(mask))
            )

        message = None
        if prompts.empty:
            message = (
                "SAM3 requires a text or box/point prompt; 'segment everything' is not "
                "supported by the Transformers image path. Provide prompts.text, "
                "prompts.points, or prompts.boxes."
            )
        return SegmentResult(masks=results, message=message)


# --------------------------------------------------------------------------- #
# Fallback backend (ultralytics MobileSAM / FastSAM, no gated weights)
# --------------------------------------------------------------------------- #
class FallbackSegmenter(Segmenter):
    """ultralytics MobileSAM/FastSAM. Points/boxes always; text via FastSAM+CLIP."""

    name = "fallback"
    text_supported = True  # via FastSAM CLIP text prompting
    EVERYTHING_CAP = 30

    def __init__(self, device: str, requested: str = "mobile_sam") -> None:
        super().__init__(device)
        self.requested = requested
        self._sam = None  # MobileSAM (points/boxes)
        self._fastsam = None  # FastSAM (text/everything)

    def load(self) -> None:
        try:
            from ultralytics import SAM  # noqa: F401
        except Exception as exc:
            raise RuntimeError(
                "Fallback backend requires `ultralytics`. Install it:\n"
                "  pip install ultralytics\n"
                f"Underlying import error: {exc}"
            ) from exc
        # Models are downloaded on first use by ultralytics; defer to _get_*.

    def _ul_device(self) -> str:
        # ultralytics accepts 'mps', 'cpu', or a cuda index.
        return self.device if self.device in ("mps", "cpu") else 0

    def _get_sam(self):
        if self._sam is None:
            from ultralytics import SAM

            self._sam = SAM("mobile_sam.pt")
        return self._sam

    def _get_fastsam(self):
        if self._fastsam is None:
            from ultralytics import FastSAM

            self._fastsam = FastSAM("FastSAM-s.pt")
        return self._fastsam

    @staticmethod
    def _masks_from_result(result, out_width: int, out_height: int) -> list[tuple[float, np.ndarray]]:
        out: list[tuple[float, np.ndarray]] = []
        if result.masks is None:
            return out
        data = result.masks.data  # tensor (n, H, W)
        confs = (
            result.boxes.conf.tolist()
            if getattr(result, "boxes", None) is not None and result.boxes is not None
            else [1.0] * len(data)
        )
        for i in range(len(data)):
            arr = data[i].detach().cpu().numpy().astype(bool)
            score = float(confs[i]) if i < len(confs) else 1.0
            out.append((score, resize_mask(arr, out_width, out_height)))
        return out

    def segment(
        self,
        image: Image.Image,
        prompts: Prompts,
        out_width: int,
        out_height: int,
        multimask: bool = False,
    ) -> SegmentResult:
        self.ensure_loaded()
        np_img = np.asarray(image.convert("RGB"))
        results: list[MaskResult] = []
        dev = self._ul_device()

        for phrase in prompts.text:
            r = self._get_fastsam()(np_img, texts=phrase, device=dev, verbose=False)[0]
            for score, mask in self._masks_from_result(r, out_width, out_height):
                results.append(
                    MaskResult(label=phrase, score=score, mask=mask, bbox=mask_to_bbox(mask))
                )

        for i, box in enumerate(prompts.boxes):
            r = self._get_sam()(np_img, bboxes=[list(box)], device=dev, verbose=False)[0]
            for score, mask in self._masks_from_result(r, out_width, out_height):
                results.append(
                    MaskResult(label=f"object_{i}", score=score, mask=mask, bbox=mask_to_bbox(mask))
                )

        fg = [p for p in prompts.points if p.label == 1]
        if fg:
            pts = [[p.x, p.y] for p in fg]
            labels = [1] * len(fg)
            r = self._get_sam()(np_img, points=pts, labels=labels, device=dev, verbose=False)[0]
            for j, (score, mask) in enumerate(self._masks_from_result(r, out_width, out_height)):
                results.append(
                    MaskResult(label=f"point_{j}", score=score, mask=mask, bbox=mask_to_bbox(mask))
                )

        message = None
        if prompts.empty:
            r = self._get_fastsam()(np_img, device=dev, verbose=False)[0]
            for j, (score, mask) in enumerate(self._masks_from_result(r, out_width, out_height)):
                if j >= self.EVERYTHING_CAP:
                    break
                results.append(
                    MaskResult(label=f"object_{j}", score=score, mask=mask, bbox=mask_to_bbox(mask))
                )

        return SegmentResult(masks=results, message=message)


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def build_segmenter(model_name: str, device: Optional[str] = None) -> Segmenter:
    device = device or detect_device()
    key = (model_name or "mock").strip().lower()
    if key == "mock":
        return MockSegmenter(device)
    if key in ("sam3", "sam3.1"):
        model_id = "facebook/sam3.1" if key == "sam3.1" else "facebook/sam3"
        return Sam3Segmenter(device, model_id=model_id)
    if key in ("mobile_sam", "mobilesam", "sam2", "fastsam", "fallback"):
        return FallbackSegmenter(device, requested=key)
    logger.warning("Unknown SAM_MODEL='%s'; falling back to mock.", model_name)
    return MockSegmenter(device)
