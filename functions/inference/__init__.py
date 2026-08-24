# ──────────────────────────────────────────────────────────────────────
# Coconut Pathology Detection Service — Inference Package
# System A: Macroscopic UAV & Spectral-Morphological Pipeline
# ──────────────────────────────────────────────────────────────────────

from inference.spectral_pipeline import (
    SpectralInferencePipeline,
    AerialSpectralPipeline,
    CanopyHotspot,
    SpectralPipelineResult,
)

__all__ = [
    "SpectralInferencePipeline",
    "AerialSpectralPipeline",
    "CanopyHotspot",
    "SpectralPipelineResult",
]
