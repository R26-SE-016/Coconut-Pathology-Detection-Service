# 🧪 Testing Guide: Coconut Pathology Detection Service (Backend)

This document provides instructions on how to execute the unit tests and NFR benchmark suites for the **Pillar 3 Python Backend Microservice** (`Coconut-Pathology-Detection-Service`).

---

## 🚀 1. How to Run the Tests

### A. Run All Unit & Algorithmic Tests
Open a terminal in the `functions/` directory and run:

```bash
cd functions
python -m unittest discover -s tests -p "test_*.py" -v
```

Alternatively, if `pytest` is installed in your virtual environment:
```bash
pytest tests/ -v
```

---

### B. Run Non-Functional Requirements (NFR) Performance Benchmarks
From the root of `Coconut-Pathology-Detection-Service/`:

```bash
python benchmarks/benchmark_nfr.py
```

---

## 🔍 2. Test Suite Breakdown & What Each Test Does

### 📁 `functions/tests/test_spectral_algorithms.py`
Tests the mathematical correctness and morphological image processing algorithms in `inference/spectral_pipeline.py` (System A UAV Aerial Diagnostics):

| Test Case | Description & Purpose |
| :--- | :--- |
| **`test_vari_formula_correctness`** | Evaluates the Visible Atmospherically Resistant Index formula $\text{VARI} = \frac{G - R}{G + R - B}$ on synthesized RGB foliage to confirm healthy vegetation yields positive values ($>0.05$). |
| **`test_vari_chlorosis_drop`** | Verifies that yellowing chlorotic coconut leaves produce a significant drop in VARI ($0.00$) compared to healthy green foliage ($0.77$). |
| **`test_ndvi_formula_bounds`** | Evaluates Normalized Difference Vegetation Index $\text{NDVI} = \frac{\text{NIR} - R}{\text{NIR} + R}$ to guarantee values are strictly bounded within $[-1.0, 1.0]$. |
| **`test_excess_green_exg`** | Verifies that the Chromatic Excess Green Index ($\text{ExG} = 2G - R - B$) accurately segments green canopy pixels ($>0$) from background bare soil ($<0$). |
| **`test_pipeline_execution_vari`** | Executes an end-to-end simulation of `SpectralInferencePipeline.process()` using VARI, verifying crown detection, canopy coverage %, purity %, and base64 colormap generation. |
| **`test_pipeline_execution_ndvi`** | Executes an end-to-end simulation of `SpectralInferencePipeline.process()` using 4-band multispectral NIR input, verifying Estate Health Grade classification. |

---

### 📁 `functions/tests/test_entropy_ood.py`
Tests the Shannon Entropy Out-of-Distribution (OOD) decision engine and Coconut Research Institute (CRI) biosecurity protocols (System B Mobile Diagnostics):

| Test Case | Description & Purpose |
| :--- | :--- |
| **`test_sharp_distribution_low_entropy`** | Verifies that high-certainty in-distribution predictions (e.g. 96% confidence) yield very low entropy ($H < 1.0\text{ bits}$), well below the OOD cutoff threshold ($H_{th} = 2.10\text{ bits}$). |
| **`test_uniform_distribution_high_entropy`** | Verifies that uncertain, uniform random distributions across 5 classes (representing non-coconut leaf images or noise) yield maximum entropy ($H = \log_2(5) \approx 2.32\text{ bits}$), exceeding the $2.10\text{ bits}$ threshold and triggering OOD rejection. |
| **`test_ood_cutoff_boundary`** | Tests borderline probability distributions to ensure mathematical stability around the $H_{th} = 2.10\text{ bits}$ decision boundary. |
| **`test_all_classes_mapped`** | Verifies that all 5 target research classes (`bud_rot`, `leaf_blight`, `stem_bleeding`, `weligama_coconut_leaf_wilt`, and `healthy`) contain complete scientific names, chemical treatments, cultural measures, and quarantine flags. |
| **`test_critical_threats_require_strict_action`** | Asserts that high-contagion pathogens (Bud Rot and Weligama Leaf Wilt) are strictly flagged as `critical` severity with mandatory quarantine/sanitation enforcement. |

---

### 📁 `benchmarks/benchmark_nfr.py`
Automated benchmark measuring Non-Functional Requirements (NFR):
1. **Spectral Latency**: Measures inference time across $256\times256$, $512\times512$, $1024\times1024$, and $2048\times2048$ resolution images over 20 iterations (Mean, P95, P99, Throughput FPS).
2. **OOD Sensitivity**: Evaluates Shannon Entropy across 5 realistic prediction scenarios (Bud Rot 98%, Leaf Blight 85%, Borderline 65%, Ambiguous 40%, Uniform Noise 20%).
3. **Model Storage Footprint**: Verifies the INT8 quantized MobileNetV2 model binary size against the mobile constraint target ($<3.0\text{ MB}$).
