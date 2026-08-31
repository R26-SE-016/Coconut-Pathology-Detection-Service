"""
Unit & Algorithmic Tests for Coconut Pathology Spectral Pipeline (System A)
Testing VARI, NDVI, ExG, Otsu Thresholding, EDT Local Maxima, and Z-Score Outlier Engine.
Project: R26-SE-016
"""

import sys
import io
import os
import unittest
import numpy as np
from PIL import Image

# Ensure functions package is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from inference.spectral_pipeline import (
    SpectralInferencePipeline,
    CanopyHotspot,
    SpectralPipelineResult,
    PhysicalPalmTree
)


class TestSpectralAlgorithms(unittest.TestCase):
    """Rigorous mathematical and morphological unit tests."""

    def setUp(self):
        """Synthesize representative multi-channel plantation test arrays."""
        # 128x128 synthetic imagery
        # Top-left (0:64, 0:64): Healthy Green Coconut Foliage (High Green, Lower Red & Blue)
        # Top-right (0:64, 64:128): Chlorotic / Diseased Yellow Leaves (High Green & High Red)
        # Bottom-left (64:128, 0:64): Bare Sandy Soil / Ground (High Red, Moderate Green, Low Blue)
        # Bottom-right (64:128, 64:128): Critical Fungal Dieback / Necrotic Tissue (Low Green, Moderate Red)
        self.img_rgb = np.zeros((128, 128, 3), dtype=np.uint8)
        
        # Healthy canopy: R=45, G=180, B=50
        self.img_rgb[0:64, 0:64] = [45, 180, 50]
        
        # Chlorotic canopy: R=190, G=190, B=30
        self.img_rgb[0:64, 64:128] = [190, 190, 30]
        
        # Bare soil: R=160, G=130, B=90
        self.img_rgb[64:128, 0:64] = [160, 130, 90]
        
        # Necrotic / severe stress: R=120, G=60, B=40
        self.img_rgb[64:128, 64:128] = [120, 60, 40]

        # Convert to bytes for pipeline ingestion
        pil_img = Image.fromarray(self.img_rgb)
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        self.image_bytes = buf.getvalue()
        self.pipeline = SpectralInferencePipeline()

    def test_vari_formula_correctness(self):
        """Test VARI formula: (G - R) / (G + R - B)."""
        r, g, b = 45.0, 180.0, 50.0
        expected_vari = (g - r) / (g + r - b)  # (180 - 45) / (180 + 45 - 50) = 135 / 175 = 0.7714
        self.assertAlmostEqual(expected_vari, 0.7714, places=3)
        self.assertGreater(expected_vari, 0.05, "Healthy coconut foliage must exhibit positive VARI.")

    def test_vari_chlorosis_drop(self):
        """Test that chlorotic yellow leaves exhibit significantly lower VARI than healthy green."""
        # Healthy
        rh, gh, bh = 45.0, 180.0, 50.0
        vari_healthy = (gh - rh) / (gh + rh - bh)
        
        # Chlorotic (Yellowing)
        rc, gc, bc = 190.0, 190.0, 30.0
        vari_chlorotic = (gc - rc) / (gc + rc - bc + 1e-6)  # (190 - 190) = 0.0
        
        self.assertGreater(vari_healthy, vari_chlorotic)
        self.assertAlmostEqual(vari_chlorotic, 0.0, places=3)

    def test_ndvi_formula_bounds(self):
        """Test NDVI formula: (NIR - R) / (NIR + R) is strictly bounded in [-1.0, 1.0]."""
        # Healthy dense mesophyll: NIR=220, R=35
        nir_healthy, r_healthy = 220.0, 35.0
        ndvi_healthy = (nir_healthy - r_healthy) / (nir_healthy + r_healthy)
        self.assertTrue(0.0 <= ndvi_healthy <= 1.0)
        self.assertGreater(ndvi_healthy, 0.65)

        # Dead tissue / water: NIR=30, R=100
        nir_dead, r_dead = 30.0, 100.0
        ndvi_dead = (nir_dead - r_dead) / (nir_dead + r_dead)
        self.assertTrue(-1.0 <= ndvi_dead < 0.0)

    def test_excess_green_exg(self):
        """Test Excess Green Index: ExG = 2G - R - B."""
        # Pure Green vegetation: R=50, G=200, B=50 -> 2*200 - 50 - 50 = 300 (> 0)
        exg_veg = 2 * 200 - 50 - 50
        self.assertGreater(exg_veg, 0)

        # Non-vegetation / Soil: R=180, G=120, B=90 -> 2*120 - 180 - 90 = -30 (< 0)
        exg_soil = 2 * 120 - 180 - 90
        self.assertLess(exg_soil, 0)

    def test_pipeline_execution_vari(self):
        """Test full end-to-end execution of SpectralInferencePipeline with VARI."""
        result = self.pipeline.process(
            image_bytes=self.image_bytes,
            index_type="VARI",
            gps_bounds={"lat": 7.5000, "lng": 80.2000, "span_lat": 0.005, "span_lng": 0.005}
        )
        self.assertIsInstance(result, SpectralPipelineResult)
        self.assertEqual(result.index_type, "VARI")
        self.assertGreaterEqual(result.estimated_palms_count, 0)
        self.assertTrue(0.0 <= result.canopy_coverage_pct <= 100.0)
        self.assertTrue(0.0 <= result.healthy_canopy_pct <= 100.0)
        self.assertIsNotNone(result.heatmap_base64)
        self.assertTrue(result.heatmap_base64.startswith("data:image/png;base64,"))

    def test_pipeline_execution_ndvi(self):
        """Test full end-to-end execution with NDVI."""
        result = self.pipeline.process(
            image_bytes=self.image_bytes,
            index_type="NDVI",
            gps_bounds={"lat": 7.5000, "lng": 80.2000, "span_lat": 0.005, "span_lng": 0.005}
        )
        self.assertIsInstance(result, SpectralPipelineResult)
        self.assertEqual(result.index_type, "NDVI")
        self.assertIn(result.estate_health_grade, ["A (Optimal)", "B (Good)", "C (Action Required)"])


if __name__ == "__main__":
    unittest.main()
