# ──────────────────────────────────────────────────────────────────────
# System A — SAHI Slicing Pipeline for YOLOv11
# Slices 4K UAV orthomosaics into 1024×1024 tiles (20 % overlap)
# and runs inference using the bundled YOLOv11 weights.
# ──────────────────────────────────────────────────────────────────────

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional

from firebase_functions import logger
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.prediction import PredictionResult


# ── Configuration ────────────────────────────────────────────────────

@dataclass(frozen=True)
class PipelineConfig:
    """Immutable configuration for the SAHI + YOLOv11 inference pipeline."""

    # Path to YOLOv11 weights (bundled with the Cloud Function deployment)
    model_path: str = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "models",
        "coconut_yolov11.pt",
    )

    # SAHI slicing parameters
    slice_width: int = 1024
    slice_height: int = 1024
    overlap_width_ratio: float = 0.20
    overlap_height_ratio: float = 0.20

    # Model parameters
    confidence_threshold: float = 0.30
    device: str = "cpu"  # Cloud Functions don't have GPU; use CPU

    # Target classes produced by the custom-trained YOLOv11
    target_classes: tuple = ("v_cut", "scorching", "wilting")


# ── Singleton model holder ───────────────────────────────────────────

_detection_model: Optional[AutoDetectionModel] = None


def _get_model(config: PipelineConfig) -> AutoDetectionModel:
    """
    Lazy-load the YOLOv11 model via SAHI's ``AutoDetectionModel``.

    The model is loaded once per Cloud Function instance (warm start)
    and reused across invocations to minimise cold-start latency.
    """
    global _detection_model

    if _detection_model is None:
        logger.info(
            f"Loading YOLOv11 model from {config.model_path} "
            f"(device={config.device})"
        )
        _detection_model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=config.model_path,
            confidence_threshold=config.confidence_threshold,
            device=config.device,
        )
        logger.info("YOLOv11 model loaded successfully.")

    return _detection_model


# ── Public API ───────────────────────────────────────────────────────

@dataclass
class DetectionResult:
    """A single bounding-box detection returned by the pipeline."""

    bbox: List[float]          # [x_min, y_min, x_max, y_max] — absolute pixels
    category_name: str         # e.g. "v_cut", "scorching", "wilting"
    category_id: int
    confidence: float          # 0.0 – 1.0


@dataclass
class PipelineOutput:
    """Aggregated output from a single orthomosaic inference run."""

    detections: List[DetectionResult] = field(default_factory=list)
    image_width: int = 0
    image_height: int = 0
    num_slices: int = 0


class SahiInferencePipeline:
    """
    Orchestrates the SAHI slicing + YOLOv11 inference pipeline.

    Usage::

        pipeline = SahiInferencePipeline()
        output   = pipeline.run("/tmp/orthomosaic_001.tif")
    """

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self.config = config or PipelineConfig()

    # ── Core ─────────────────────────────────────────────────────────

    def run(self, image_path: str) -> PipelineOutput:
        """
        Run sliced inference on a single orthomosaic image.

        Parameters
        ----------
        image_path : str
            Absolute path to the downloaded orthomosaic in ``/tmp``.

        Returns
        -------
        PipelineOutput
            Parsed detections ready for downstream NMS merging.
        """
        model = _get_model(self.config)

        logger.info(
            f"Starting SAHI sliced prediction on {image_path} "
            f"(slice={self.config.slice_width}×{self.config.slice_height}, "
            f"overlap={self.config.overlap_width_ratio})"
        )

        prediction_result: PredictionResult = get_sliced_prediction(
            image=image_path,
            detection_model=model,
            slice_height=self.config.slice_height,
            slice_width=self.config.slice_width,
            overlap_height_ratio=self.config.overlap_height_ratio,
            overlap_width_ratio=self.config.overlap_width_ratio,
            verbose=0,
        )

        # Parse SAHI ObjectPrediction objects → DetectionResult DTOs
        detections: List[DetectionResult] = []
        for obj_pred in prediction_result.object_prediction_list:
            bbox = obj_pred.bbox.to_xyxy()  # [x1, y1, x2, y2]
            detections.append(
                DetectionResult(
                    bbox=[float(b) for b in bbox],
                    category_name=obj_pred.category.name,
                    category_id=obj_pred.category.id,
                    confidence=float(obj_pred.score.value),
                )
            )

        logger.info(
            f"SAHI inference complete — {len(detections)} raw detections "
            f"across tiles."
        )

        return PipelineOutput(
            detections=detections,
            image_width=prediction_result.image_width,
            image_height=prediction_result.image_height,
            num_slices=len(prediction_result.object_prediction_list),
        )
