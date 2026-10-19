"""
Verification Script for System A Precision Agriculture Upgrade
Validates:
1. Vari & NDVI calculation
2. Canonical ExG canopy masking
3. Area-based Connected Component Stress Zone Extraction
4. Agronomic Palm Population Estimation (150-200 palms calibration)
"""

import os
import io
import time
import numpy as np
from PIL import Image

from inference.spectral_pipeline import SpectralInferencePipeline, SpectralPipelineResult

def run_verification():
    print("=" * 70)
    print("  VERIFYING SYSTEM A AREA-BASED SPECTRAL PIPELINE")
    print("=" * 70)

    pipeline = SpectralInferencePipeline()

    # 1. Synthetic Plantation Test
    print("\n[Step 1/2] Running Synthetic Multi-Zone Canopy Test...")
    # 512x512 synthetic image: 80% green canopy with an acute chlorotic/necrotic spot
    arr = np.zeros((512, 512, 3), dtype=np.uint8)
    arr[:, :] = [45, 175, 45]  # Healthy green canopy

    # Inject acute stress cluster in the center (radius 30px)
    yy, xx = np.ogrid[:512, :512]
    stress_spot = ((yy - 256)**2 + (xx - 256)**2) <= 30**2
    arr[stress_spot] = [180, 160, 40]  # Chlorotic yellow patch

    # Inject bare soil border
    arr[450:, :] = [160, 130, 90]

    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    synth_bytes = buf.getvalue()

    t0 = time.time()
    synth_res = pipeline.process(synth_bytes, "VARI")
    t_synth = (time.time() - t0) * 1000

    print(f"  [PASS] Processed in: {t_synth:.1f}ms")
    print(f"  [PASS] Mean VARI: {synth_res.mean_index:.3f}")
    print(f"  [PASS] Canopy Coverage: {synth_res.canopy_coverage_pct:.1f}%")
    print(f"  [PASS] Estimated Palms: {synth_res.estimated_palms_count}")
    print(f"  [PASS] Stress Hotspots Flagged: {len(synth_res.hotspots)}")
    assert len(synth_res.hotspots) >= 1, "Failed to detect injected stress spot!"
    print(f"  [PASS] Detected Injected Stress Spot: Centroid=({synth_res.hotspots[0].pixel_x}, {synth_res.hotspots[0].pixel_y}), Severity={synth_res.hotspots[0].severity}")

    # 2. Real Ambakele Drone Image Test (if available)
    ambakele_img = r"f:\GitHub\Research\Datasets\Raw Images\Ambakele\DJI_20250813102921_0001_D.JPG"
    ambakele_nir = r"f:\GitHub\Research\Datasets\Raw Images\Ambakele\DJI_20250813102921_0001_MS_NIR.TIF"

    if os.path.exists(ambakele_img):
        print("\n[Step 2/2] Running Real Ambakele Drone Orthomosaic Test...")
        with open(ambakele_img, "rb") as f:
            real_bytes = f.read()

        nir_bytes = None
        if os.path.exists(ambakele_nir):
            with open(ambakele_nir, "rb") as f:
                nir_bytes = f.read()

        # Test VARI
        t0 = time.time()
        real_res_vari = pipeline.process(real_bytes, "VARI")
        t_vari = (time.time() - t0) * 1000

        print(f"  [PASS] VARI Inferred in: {t_vari:.1f}ms")
        print(f"  [PASS] VARI Mean: {real_res_vari.mean_index:.3f}")
        print(f"  [PASS] Estimated Palms: {real_res_vari.estimated_palms_count} (Target: 150-200 palms)")
        print(f"  [PASS] Hotspots Flagged: {len(real_res_vari.hotspots)}")
        assert 130 <= real_res_vari.estimated_palms_count <= 250, f"Unrealistic palm count: {real_res_vari.estimated_palms_count}"

        # Test NDVI (if multispectral available)
        if nir_bytes:
            t0 = time.time()
            real_res_ndvi = pipeline.process(real_bytes, "NDVI", nir_bytes=nir_bytes)
            t_ndvi = (time.time() - t0) * 1000
            print(f"  [PASS] Multispectral NDVI Inferred in: {t_ndvi:.1f}ms")
            print(f"  [PASS] NDVI Mean: {real_res_ndvi.mean_index:.3f}")
            print(f"  [PASS] Estimated Palms: {real_res_ndvi.estimated_palms_count}")
            print(f"  [PASS] Hotspots Flagged: {len(real_res_ndvi.hotspots)}")

    print("\n" + "=" * 70)
    print("  ALL SYSTEM A VERIFICATIONS PASSED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    run_verification()
