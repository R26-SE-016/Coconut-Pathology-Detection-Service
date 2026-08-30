# ══════════════════════════════════════════════════════════════════════
# Coconut Pathology Detection Service — Unified Spectral & Morphological Pipeline
# Project: R26-SE-016 — Multiscale Computer Vision Ecosystem
#
# 1. Strategy A: Canonical Chromatic Excess-Green Canopy Masking
# 2. Strategy B: Euclidean Distance Transform Local Maxima (EDT) for
#    deterministic, camera-invariant individual physical tree crown extraction.
# 3. Strategy C: Discrete-Tree Local Moving-Window Z-Score Anomaly Engine.
# ══════════════════════════════════════════════════════════════════════

from __future__ import annotations

import base64
import gc
import io
import math
import os
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageFile
from scipy.ndimage import distance_transform_edt, gaussian_filter, maximum_filter

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


@dataclass
class PhysicalPalmTree:
    """Represents a discrete physical coconut palm tree in the plantation."""
    tree_id: int
    cx: int
    cy: int
    lat: float
    lng: float
    crown_radius_px: int
    crown_area_px: int


@dataclass
class CanopyHotspot:
    id: str
    lat: float
    lng: float
    pixel_x: int
    pixel_y: int
    mean_index_value: float
    severity: str  # "critical" | "high" | "moderate"
    area_sq_pixels: int
    radius_meters: float
    recommended_action: str
    z_score: Optional[float] = None
    relative_drop_pct: Optional[float] = None
    status: str = "pending"  # "pending" | "inspected" | "resolved"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": str(self.id),
            "location": {"lat": float(self.lat), "lng": float(self.lng)},
            "pixel_coordinates": {"x": int(self.pixel_x), "y": int(self.pixel_y)},
            "mean_index_value": float(round(float(self.mean_index_value), 4)),
            "severity": str(self.severity),
            "area_sq_pixels": int(self.area_sq_pixels),
            "radius_meters": float(round(float(self.radius_meters), 2)),
            "recommended_action": str(self.recommended_action),
            "z_score": float(round(float(self.z_score), 2)) if self.z_score is not None else None,
            "relative_drop_pct": float(round(float(self.relative_drop_pct), 1)) if self.relative_drop_pct is not None else None,
            "status": str(self.status),
        }


@dataclass
class SpectralPipelineResult:
    index_type: str  # "NDVI" | "VARI"
    image_width: int
    image_height: int
    mean_index: float
    min_index: float
    max_index: float
    canopy_coverage_pct: float
    ground_exposure_pct: float
    healthy_canopy_pct: float
    moderate_stress_pct: float
    severe_stress_pct: float
    estate_health_grade: str
    pathology_risk_index: str
    estimated_palms_count: int
    healthy_palms_count: int
    at_risk_palms_count: int
    heatmap_base64: str
    hotspots: List[CanopyHotspot] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index_type": str(self.index_type),
            "image_dimensions": {
                "width": int(self.image_width),
                "height": int(self.image_height),
            },
            "statistics": {
                "mean_index": float(round(float(self.mean_index), 4)),
                "min_index": float(round(float(self.min_index), 4)),
                "max_index": float(round(float(self.max_index), 4)),
                "canopy_coverage_pct": float(round(float(self.canopy_coverage_pct), 2)),
                "ground_exposure_pct": float(round(float(self.ground_exposure_pct), 2)),
                "healthy_canopy_pct": float(round(float(self.healthy_canopy_pct), 2)),
                "moderate_stress_pct": float(round(float(self.moderate_stress_pct), 2)),
                "severe_stress_pct": float(round(float(self.severe_stress_pct), 2)),
                "estate_health_grade": str(self.estate_health_grade),
                "pathology_risk_index": str(self.pathology_risk_index),
                "estimated_palms_count": int(self.estimated_palms_count),
                "healthy_palms_count": int(self.healthy_palms_count),
                "at_risk_palms_count": int(self.at_risk_palms_count),
            },
            "heatmap_base64": str(self.heatmap_base64),
            "hotspots": [h.to_dict() for h in self.hotspots],
        }


class SpectralInferencePipeline:
    """
    Unified Dual-Index (NDVI & VARI) processor for Macroscopic Aerial Surveillance.
    Guarantees 100% deterministic physical tree counts and locations across both modes.
    """

    def __init__(self, epsilon: float = 1e-7):
        self.epsilon = epsilon

    def _apply_rdylgn_colormap(
        self, index_norm: np.ndarray, canopy_mask: np.ndarray
    ) -> np.ndarray:
        """
        Applies a high-contrast Red-Yellow-Green (RdYlGn) gradient:
        - 0.0 (Severe stress): Red   [215, 48, 39]
        - 0.5 (Moderate stress): Yellow [254, 224, 139]
        - 1.0 (Vigorous canopy): Green [26, 152, 80]
        - Non-canopy (soil/background): Muted dark slate [15, 23, 42]
        """
        h, w = index_norm.shape
        rgb_heatmap = np.zeros((h, w, 3), dtype=np.uint8)

        c_red = np.array([215, 48, 39], dtype=np.float32)
        c_yellow = np.array([254, 224, 139], dtype=np.float32)
        c_green = np.array([26, 152, 80], dtype=np.float32)
        c_soil = np.array([15, 23, 42], dtype=np.uint8)

        lower_mask = (index_norm <= 0.5) & canopy_mask
        if np.any(lower_mask):
            t_low = (index_norm[lower_mask] / 0.5)[:, np.newaxis]
            rgb_heatmap[lower_mask] = np.clip(
                (1.0 - t_low) * c_red + t_low * c_yellow, 0, 255
            ).astype(np.uint8)

        upper_mask = (index_norm > 0.5) & canopy_mask
        if np.any(upper_mask):
            t_high = ((index_norm[upper_mask] - 0.5) / 0.5)[:, np.newaxis]
            rgb_heatmap[upper_mask] = np.clip(
                (1.0 - t_high) * c_yellow + t_high * c_green, 0, 255
            ).astype(np.uint8)

        soil_mask = ~canopy_mask
        rgb_heatmap[soil_mask] = c_soil

        return rgb_heatmap

    def _detect_physical_trees(
        self,
        canopy_mask: np.ndarray,
        gps_bounds: Optional[Dict[str, float]] = None,
    ) -> List[PhysicalPalmTree]:
        """
        Deterministic Strategy B:
        Applies Euclidean Distance Transform (EDT) and Local Maxima Peak Extraction
        on the canonical physical canopy mask.
        Tree spacing is fixed to the physical agronomic planting grid (~8m = ~26px at 1024 scale).
        """
        h, w = canopy_mask.shape

        edt = distance_transform_edt(canopy_mask)
        if not np.any(edt > 0):
            return []

        # Smooth to eliminate micro-frond noise and isolate single trunk apices
        smoothed_edt = gaussian_filter(edt, sigma=2.2)

        # Standard 8m mature coconut palm planting grid (~25-28px at 1024 scale)
        min_tree_dist_px = max(24, int(min(h, w) / 36))
        footprint_size = 2 * min_tree_dist_px + 1

        # Extract local maxima peaks
        local_max = maximum_filter(smoothed_edt, size=footprint_size) == smoothed_edt
        
        # Valid coconut trunk apex must have minimum crown depth (>= 6px inside canopy)
        valid_peaks = local_max & canopy_mask & (smoothed_edt >= 6.0)
        peak_ys, peak_xs = np.where(valid_peaks)

        base_lat = gps_bounds.get("lat", 7.2906) if gps_bounds else 7.2906
        base_lng = gps_bounds.get("lng", 80.6337) if gps_bounds else 80.6337
        span_lat = gps_bounds.get("span_lat", 0.005) if gps_bounds else 0.005
        span_lng = gps_bounds.get("span_lng", 0.005) if gps_bounds else 0.005

        trees: List[PhysicalPalmTree] = []
        for idx, (cy, cx) in enumerate(zip(peak_ys, peak_xs)):
            lat = base_lat + (0.5 - (cy / h)) * span_lat
            lng = base_lng + ((cx / w) - 0.5) * span_lng
            crown_r = max(14, int(edt[cy, cx]))

            trees.append(
                PhysicalPalmTree(
                    tree_id=idx + 1,
                    cx=int(cx),
                    cy=int(cy),
                    lat=round(lat, 6),
                    lng=round(lng, 6),
                    crown_radius_px=crown_r,
                    crown_area_px=int(math.pi * (crown_r ** 2)),
                )
            )

        return trees

    def _profile_tree_anomalies(
        self,
        trees: List[PhysicalPalmTree],
        index_array: np.ndarray,
        canopy_mask: np.ndarray,
        index_type: str,
        max_hotspots: int = 25,
    ) -> List[CanopyHotspot]:
        """
        Strategy C:
        Evaluates the specific spectral index (NDVI or VARI) across each detected
        physical tree and computes moving-window relative Z-Scores against neighboring trees.
        """
        if not trees:
            return []

        h, w = index_array.shape
        tree_scores: List[float] = []

        # Sample spectral index per individual tree disk
        for t in trees:
            cx, cy, r = t.cx, t.cy, t.crown_radius_px
            y0, y1 = max(0, cy - r), min(h, cy + r + 1)
            x0, x1 = max(0, cx - r), min(w, cx + r + 1)

            yy, xx = np.ogrid[y0:y1, x0:x1]
            dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
            disk = (dist <= r) & canopy_mask[y0:y1, x0:x1]

            if np.any(disk):
                score = float(np.mean(index_array[y0:y1, x0:x1][disk]))
            else:
                score = float(index_array[cy, cx])
            tree_scores.append(score)

        # Neighbor spatial search radius (~35m = ~3x tree spacing)
        neighbor_dist_thresh = max(60, int(min(h, w) / 12))

        is_ndvi = (index_type == "NDVI")
        healthy_cutoff = 0.52 if is_ndvi else 0.06
        severe_cutoff = 0.36 if is_ndvi else -0.01

        hotspots: List[CanopyHotspot] = []

        for i, t in enumerate(trees):
            t_score = tree_scores[i]

            # Find nearest neighbor palms
            neighbors = []
            for j, other in enumerate(trees):
                if i == j:
                    continue
                d = math.hypot(t.cx - other.cx, t.cy - other.cy)
                if d <= neighbor_dist_thresh:
                    neighbors.append(tree_scores[j])

            if len(neighbors) >= 2:
                local_mean = float(np.mean(neighbors))
                local_std = float(np.std(neighbors))
            else:
                local_mean = t_score
                local_std = 0.04

            z_score = (t_score - local_mean) / max(local_std, 0.03)
            rel_drop = max(0.0, ((local_mean - t_score) / (abs(local_mean) + self.epsilon)) * 100.0)

            # Anomaly criteria
            is_anomaly = (z_score <= -1.25 and t_score < healthy_cutoff) or (t_score < severe_cutoff)

            if is_anomaly:
                if z_score <= -2.0 or t_score < severe_cutoff:
                    sev = "critical"
                    rec = f"Acute localized anomaly (Tree #{t.tree_id}, Z={z_score:.2f}, -{rel_drop:.0f}% vs neighbors). Priority ground scan for Bud Rot / Stem Bleeding."
                else:
                    sev = "high"
                    rec = f"Localized chlorosis outlier (Tree #{t.tree_id}, Z={z_score:.2f}, -{rel_drop:.0f}% vs neighbors). Inspect for crown mite infestation or localized root decay."

                hs = CanopyHotspot(
                    id=f"tree_{t.tree_id}_{uuid.uuid4().hex[:6]}",
                    lat=t.lat,
                    lng=t.lng,
                    pixel_x=t.cx,
                    pixel_y=t.cy,
                    mean_index_value=round(t_score, 3),
                    severity=sev,
                    area_sq_pixels=t.crown_area_px,
                    radius_meters=max(3.5, round(t.crown_radius_px * 0.28, 1)),
                    recommended_action=rec,
                    z_score=round(z_score, 2),
                    relative_drop_pct=round(rel_drop, 1),
                )
                hotspots.append(hs)

        # Sort lowest Z-score (worst drop) first
        hotspots.sort(key=lambda x: (x.z_score if x.z_score is not None else 0.0, x.mean_index_value))
        return hotspots[:max_hotspots]

    def process(
        self,
        image_bytes: bytes,
        index_type: str = "VARI",
        nir_bytes: Optional[bytes] = None,
        gps_bounds: Optional[Dict[str, float]] = None,
    ) -> SpectralPipelineResult:
        """
        Executes the unified spectral calculation and deterministic physical tree detection.
        """
        img = Image.open(io.BytesIO(image_bytes))

        # Standard 1024px scale
        MAX_DIM = 1024
        if max(img.width, img.height) > MAX_DIM:
            scale = MAX_DIM / max(img.width, img.height)
            new_w = max(1, int(img.width * scale))
            new_h = max(1, int(img.height * scale))
            img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)

        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGB")

        img_array = np.array(img, dtype=np.float32)

        # Strategy A: Canonical Physical Coconut Canopy Mask (RGB Chromatic)
        if img_array.ndim == 3 and img_array.shape[2] >= 3:
            r_chan = img_array[:, :, 0]
            g_chan = img_array[:, :, 1]
            b_chan = img_array[:, :, 2]

            rgb_sum = r_chan + g_chan + b_chan + self.epsilon
            norm_exg = (2.0 * g_chan - r_chan - b_chan) / rgb_sum
            gcc = g_chan / rgb_sum

            # Canonical physical canopy mask — Identical across all spectral modes
            canopy_mask = (gcc >= 0.350) & (norm_exg >= 0.045) & (g_chan > 28) & (r_chan < 235)
            del rgb_sum, norm_exg, gcc
        else:
            canopy_mask = np.ones((img.height, img.width), dtype=bool)

        # Strategy B: Deterministic Physical Tree Detection on canonical canopy
        physical_trees = self._detect_physical_trees(canopy_mask, gps_bounds)
        exact_palm_count = len(physical_trees)

        index_type = index_type.upper()

        if index_type == "NDVI":
            if img_array.ndim == 3 and img_array.shape[2] >= 4:
                red = img_array[:, :, 0]
                nir = img_array[:, :, 3]
            elif nir_bytes is not None:
                nir_img = Image.open(io.BytesIO(nir_bytes)).convert("L")
                nir_img = nir_img.resize(
                    (img.width, img.height), Image.Resampling.BILINEAR
                )
                nir = np.array(nir_img, dtype=np.float32)
                red = img_array[:, :, 0] if img_array.ndim == 3 else img_array
            else:
                red = img_array[:, :, 0] if img_array.ndim == 3 else img_array
                green = img_array[:, :, 1] if img_array.ndim == 3 else img_array * 0.8
                nir = np.clip(2.0 * green - 0.5 * red, 0, 255)

            raw_index = (nir - red) / (nir + red + self.epsilon)
            np.clip(raw_index, -1.0, 1.0, out=raw_index)

            norm_index = np.clip((raw_index - 0.25) / (0.80 - 0.25 + self.epsilon), 0.0, 1.0)
            healthy_mask = (raw_index >= 0.45) & canopy_mask
            moderate_mask = (raw_index >= 0.35) & (raw_index < 0.45) & canopy_mask
            severe_mask = (raw_index < 0.35) & canopy_mask

        else:
            if img_array.ndim == 3 and img_array.shape[2] >= 3:
                red = img_array[:, :, 0]
                green = img_array[:, :, 1]
                blue = img_array[:, :, 2]
            else:
                raise ValueError("VARI calculation requires a 3-channel RGB image.")

            raw_index = (green - red) / (green + red - blue + self.epsilon)
            np.clip(raw_index, -1.0, 1.0, out=raw_index)

            norm_index = np.clip((raw_index - (-0.02)) / (0.35 - (-0.02) + self.epsilon), 0.0, 1.0)
            healthy_mask = (raw_index >= 0.04) & canopy_mask
            moderate_mask = (raw_index >= 0.00) & (raw_index < 0.04) & canopy_mask
            severe_mask = (raw_index < 0.00) & canopy_mask

        # Strategy C: Profile anomalies on the exact physical trees
        hotspots = self._profile_tree_anomalies(
            physical_trees, raw_index, canopy_mask, index_type
        )

        # Statistics computation
        total_pixels = raw_index.size
        canopy_pixels = int(np.sum(canopy_mask))
        if canopy_pixels == 0:
            canopy_mask = np.ones_like(raw_index, dtype=bool)
            canopy_pixels = total_pixels

        canopy_values = raw_index[canopy_mask]
        mean_val = float(np.mean(canopy_values))
        min_val = float(np.min(canopy_values))
        max_val = float(np.max(canopy_values))

        coverage_pct = (canopy_pixels / total_pixels) * 100.0
        ground_exposure = max(0.0, 100.0 - coverage_pct)
        healthy_pct = (int(np.sum(healthy_mask)) / canopy_pixels) * 100.0
        moderate_pct = (int(np.sum(moderate_mask)) / canopy_pixels) * 100.0
        severe_pct = (int(np.sum(severe_mask)) / canopy_pixels) * 100.0

        at_risk_palms = len(hotspots)
        healthy_palms = max(0, exact_palm_count - at_risk_palms)

        # Biosecurity Estate Health Grade
        outlier_ratio = at_risk_palms / max(1, exact_palm_count)
        if outlier_ratio <= 0.04:
            estate_grade = "A (Optimal)"
            risk_index = "Low / Healthy"
        elif outlier_ratio <= 0.12:
            estate_grade = "B (Good)"
            risk_index = "Isolated Outliers"
        else:
            estate_grade = "C (Action Required)"
            risk_index = "Cluster Anomaly Alert"

        # Generate Colormapped Heatmap
        rgb_heatmap = self._apply_rdylgn_colormap(norm_index, canopy_mask)
        heatmap_img = Image.fromarray(rgb_heatmap)

        buffered = io.BytesIO()
        heatmap_img.save(buffered, format="PNG")
        heatmap_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # Cleanup memory
        del img_array, rgb_heatmap, norm_index
        gc.collect()

        return SpectralPipelineResult(
            index_type=index_type,
            image_width=img.width,
            image_height=img.height,
            mean_index=mean_val,
            min_index=min_val,
            max_index=max_val,
            canopy_coverage_pct=coverage_pct,
            ground_exposure_pct=ground_exposure,
            healthy_canopy_pct=healthy_pct,
            moderate_stress_pct=moderate_pct,
            severe_stress_pct=severe_pct,
            estate_health_grade=estate_grade,
            pathology_risk_index=risk_index,
            estimated_palms_count=exact_palm_count,
            healthy_palms_count=healthy_palms,
            at_risk_palms_count=at_risk_palms,
            heatmap_base64=f"data:image/png;base64,{heatmap_b64}",
            hotspots=hotspots,
        )


# Backward-compatible alias
AerialSpectralPipeline = SpectralInferencePipeline
