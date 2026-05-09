# ──────────────────────────────────────────────────────────────────────
# System A — Cross-Tile Non-Maximum Suppression (NMS)
# Merges overlapping detections from SAHI tiles into a unified
# pathological heatmap, eliminating duplicate bounding boxes that
# span tile boundaries.
# ──────────────────────────────────────────────────────────────────────

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
from firebase_functions import logger

from inference.sahi_pipeline import DetectionResult


# ── Configuration ────────────────────────────────────────────────────

@dataclass(frozen=True)
class NMSConfig:
    """Parameters for the cross-tile NMS merge pass."""

    iou_threshold: float = 0.45     # IoU above which two boxes are considered duplicates
    score_threshold: float = 0.30   # Minimum confidence to keep a detection


# ── DTOs ─────────────────────────────────────────────────────────────

@dataclass
class MergedDetection:
    """A single detection after NMS de-duplication."""

    bbox: List[float]          # [x_min, y_min, x_max, y_max]
    category_name: str
    category_id: int
    confidence: float


@dataclass
class ClassSummary:
    """Per-class aggregate statistics."""

    count: int = 0
    mean_confidence: float = 0.0
    max_confidence: float = 0.0


@dataclass
class HeatmapPayload:
    """
    Final output written to the ``heatmaps`` Firestore collection.
    """

    detections: List[MergedDetection] = field(default_factory=list)
    summary: Dict[str, ClassSummary] = field(default_factory=dict)
    total_detections: int = 0


# ── NMS Implementation ───────────────────────────────────────────────

class CrossTileNMS:
    """
    Applies per-class Non-Maximum Suppression across all tile results.

    Usage::

        nms  = CrossTileNMS()
        hmap = nms.merge(pipeline_output.detections)
    """

    def __init__(self, config: NMSConfig | None = None) -> None:
        self.config = config or NMSConfig()

    # ── Public ───────────────────────────────────────────────────────

    def merge(self, detections: List[DetectionResult]) -> HeatmapPayload:
        """
        Merge raw tile-level detections into a de-duplicated heatmap.

        1. Group detections by ``category_name``.
        2. Apply IoU-based NMS within each class.
        3. Aggregate summary statistics.
        """
        if not detections:
            logger.warn("NMS received zero detections — returning empty heatmap.")
            return HeatmapPayload()

        # Group by class
        class_groups: Dict[str, List[DetectionResult]] = {}
        for det in detections:
            class_groups.setdefault(det.category_name, []).append(det)

        merged: List[MergedDetection] = []
        summary: Dict[str, ClassSummary] = {}

        for cls_name, cls_dets in class_groups.items():
            kept = self._nms_for_class(cls_dets)
            merged.extend(kept)

            confidences = [d.confidence for d in kept]
            summary[cls_name] = ClassSummary(
                count=len(kept),
                mean_confidence=round(float(np.mean(confidences)), 4) if confidences else 0.0,
                max_confidence=round(float(np.max(confidences)), 4) if confidences else 0.0,
            )

        logger.info(
            f"NMS complete — {len(merged)} detections kept from "
            f"{len(detections)} raw inputs. "
            f"Classes: {list(summary.keys())}"
        )

        return HeatmapPayload(
            detections=merged,
            summary=summary,
            total_detections=len(merged),
        )

    # ── Private ──────────────────────────────────────────────────────

    def _nms_for_class(
        self, detections: List[DetectionResult]
    ) -> List[MergedDetection]:
        """
        Standard greedy NMS for a single class.

        Sorts by confidence descending, then iteratively suppresses boxes
        whose IoU with a higher-scoring box exceeds the threshold.
        """
        if not detections:
            return []

        # Sort descending by confidence
        sorted_dets = sorted(detections, key=lambda d: d.confidence, reverse=True)

        boxes = np.array([d.bbox for d in sorted_dets], dtype=np.float64)
        scores = np.array([d.confidence for d in sorted_dets], dtype=np.float64)

        keep_indices: List[int] = []

        while len(boxes) > 0:
            # Pick the box with the highest score
            keep_indices.append(0)
            if len(boxes) == 1:
                break

            # Compute IoU of the picked box against all remaining
            ious = self._compute_iou(boxes[0], boxes[1:])

            # Keep boxes whose IoU with the picked box is below the threshold
            mask = ious < self.config.iou_threshold
            # +1 offset because we compared against boxes[1:]
            remaining = np.where(mask)[0] + 1

            boxes = boxes[remaining]
            scores = scores[remaining]
            sorted_dets = [sorted_dets[i] for i in remaining]

        # Map kept detections → MergedDetection
        kept_dets = sorted(
            detections, key=lambda d: d.confidence, reverse=True
        )
        result: List[MergedDetection] = []
        for idx in keep_indices:
            if idx < len(kept_dets):
                d = kept_dets[idx]
                result.append(
                    MergedDetection(
                        bbox=d.bbox,
                        category_name=d.category_name,
                        category_id=d.category_id,
                        confidence=d.confidence,
                    )
                )

        return result

    @staticmethod
    def _compute_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """
        Compute IoU between a single box and an array of boxes.

        Parameters
        ----------
        box : np.ndarray, shape (4,)
            Reference box ``[x1, y1, x2, y2]``.
        boxes : np.ndarray, shape (N, 4)
            Candidate boxes.

        Returns
        -------
        np.ndarray, shape (N,)
            IoU values.
        """
        # Intersection coordinates
        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])

        inter_area = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)

        # Areas
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        union_area = box_area + boxes_area - inter_area

        # Avoid division by zero
        iou = np.where(union_area > 0, inter_area / union_area, 0.0)
        return iou
