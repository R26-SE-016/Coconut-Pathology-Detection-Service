<div align="center">

<img src="./docs/brand/logo-text.png" alt="සරුපොල් (SaruPol)" width="380" />
<br/>
<img src="./docs/brand/logo-icon.png" alt="SaruPol Icon" width="80" />

### 🥥 සරුපොල් (SaruPol) — Coconut Pathology Detection Service
**Serverless Python Multiscale Computer Vision Backend for Aerial Spectral Surveillance & Edge AI Pathology**

[![Python](https://img.shields.io/badge/Python-3.11-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Firebase](https://img.shields.io/badge/Firebase%20Functions-Gen%202-FFCA28.svg?logo=firebase&logoColor=black)](https://firebase.google.com/)
[![Google Cloud](https://img.shields.io/badge/Google%20Cloud-asia--south1-4285F4.svg?logo=google-cloud&logoColor=white)](https://cloud.google.com/)
[![Firestore](https://img.shields.io/badge/Firestore-Native%20Mode-FFCA28.svg?logo=firebase&logoColor=black)](https://firebase.google.com/docs/firestore)
[![TensorFlow Lite](https://img.shields.io/badge/Model-MobileNetV2--INT8-FF6F00.svg?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/lite)
[![Status](https://img.shields.io/badge/Status-Production%20Live-brightgreen.svg)]()

</div>

---

## 📖 Overview

**Coconut-Pathology-Detection-Service** (Project **R26-SE-016**) is the serverless AI/ML computation engine of the **SaruPol Smart Coconut Plantation Ecosystem**. 

Built on **Google Cloud Functions (Gen 2)**, **Google Cloud Firestore**, and **Google Cloud Storage** in the `asia-south1` (Mumbai) region, this service operationalizes two independent, complementary computer vision pipelines:

1. **System A — Macroscopic UAV Aerial Surveillance Engine**:
   - **Excess Green (ExG) Canopy Segmentation**: Isolates living palm crowns from inter-row soil and ground noise.
   - **Dual-Index Spectral Analysis**: Computes **VARI** (Visible Atmospherically Resistant Index) for standard RGB drone flights and **NDVI** (Normalized Difference Vegetation Index) for 4-band / NIR multispectral surveys.
   - **Euclidean Distance Transform (EDT) Local Maxima**: Deterministically extracts individual physical palm crowns and spatial geometries.
   - **Discrete Moving-Window Z-Score Outlier Engine**: Identifies biologically stressed hotspot trees ($Z \le -2.0\sigma$) and generates field dispatch tickets.

2. **System B — Microscopic On-Device Diagnostic & Sync Engine**:
   - High-throughput asynchronous **BulkWriter Firestore batch ingestion** for edge field scans.
   - User diagnostic history tracking, feedback logging, and Sri Lanka Coconut Research Institute (CRI) clinical metadata enrichment.

---

## 🏛️ System Architecture

```
                                  ┌─────────────────────────────────────────────────────────┐
                                  │      SaruPol API Gateway / Mobile App / Web UI          │
                                  └────────────────────────────┬────────────────────────────┘
                                                               │
                                ┌──────────────────────────────┴──────────────────────────────┐
                                │                                                             │
                                ▼ (Port 5001 / Cloud Functions)                               ▼ (Port 5001 / Cloud Functions)
┌─────────────────────────────────────────────────────────────┐ ┌─────────────────────────────────────────────────────────────┐
│                 System A (UAV Surveillance)                 │ │                 System B (Mobile Diagnostics)               │
│                                                             │ │                                                             │
│  ┌────────────────────────┐   ┌──────────────────────────┐  │ │  ┌────────────────────────┐   ┌──────────────────────────┐  │
│  │ process_aerial_        │   │ on_orthomosaic_          │  │ │  │ predict_mobile_disease │   │ sync_mobile_diagnostics  │  │
│  │ spectral (HTTP POST)   │   │ uploaded (Storage Event) │  │ │  │ (TFLite Fallback API)  │   │ (BulkWriter Firestore)   │  │
│  └───────────┬────────────┘   └─────────────┬────────────┘  │ │  └───────────┬────────────┘   └─────────────┬────────────┘  │
│              │                              │               │ │              │                              │               │
│              ▼                              ▼               │ │              ▼                              ▼               │
│  ┌────────────────────────┐   ┌──────────────────────────┐  │ │  ┌────────────────────────┐   ┌──────────────────────────┐  │
│  │ ExG + EDT + Z-Score    │   │ SAHI Slicing (1024x1024) │  │ │  │ MobileNetV2-INT8       │   │ Firestore batch writes   │  │
│  │ Spectral Pipeline      │   │ + YOLOv11 + Tile NMS     │  │ │  │ + Softmax Probabilities│   │ (diagnostics/ collection)│  │
│  └───────────┬────────────┘   └─────────────┬────────────┘  │ │  └───────────┬────────────┘   └─────────────┬────────────┘  │
└──────────────┼──────────────────────────────┼───────────────┘ └──────────────┼──────────────────────────────┼───────────────┘
               │                              │                                │                              │
               ▼                              ▼                                ▼                              ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                           Google Cloud Firestore & Storage Layer                                            │
│                                                                                                                             │
│       • `diagnostics/`       — Georeferenced field scans with pathogen severity & confidence metadata                       │
│       • `canopy_hotspots/`   — Discrete physiological stress outlier tickets dispatched for field inspection                │
│       • `heatmaps/`          — Colormapped vegetation index rasters and NMS bounding box metadata                           │
│       • `estates/`           — Estate boundary polygons and historical flight telemetry                                     │
│       • `users/`             — Role-based access control (Field Officers, Estate Managers, Agronomists)                     │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🧩 Cloud Functions Endpoint Reference

| Function Name | Trigger | Method | Purpose |
| :--- | :--- | :--- | :--- |
| `process_aerial_spectral` | HTTPS | `POST` | Processes raw drone orthomosaics, generates VARI/NDVI rasters, extracts palm crowns, and flags Z-score outliers |
| `get_canopy_hotspots` | HTTPS | `GET` | Retrieves flagged physiological stress hotspots for an estate |
| `update_hotspot_status` | HTTPS | `PATCH` | Updates hotspot status (`pending` $\rightarrow$ `inspected` $\rightarrow$ `resolved`) with leaf diagnostic ID |
| `sync_mobile_diagnostics` | HTTPS | `POST` | High-throughput batch synchronization of mobile leaf diagnostic records |
| `get_diagnostic_history` | HTTPS | `GET` | Fetches historical diagnostic records filtered by user or estate |
| `on_orthomosaic_uploaded` | Cloud Storage | `Event` | Automatically triggered on 4K orthomosaic `.tif` uploads $\rightarrow$ SAHI + YOLOv11 inference |
| `get_estate_heatmap` | HTTPS | `GET` | Fetches raster heatmaps and aggregated health scores for an estate |

---

## 📂 Project Structure

```
Coconut-Pathology-Detection-Service/
├── functions/
│   ├── inference/
│   │   ├── __init__.py
│   │   └── spectral_pipeline.py          # Unified ExG, VARI/NDVI, EDT, & Z-Score anomaly engine
│   ├── models/
│   │   └── system_b/
│   │       └── system_b_baseline_int8.tflite # Quantized MobileNetV2-INT8 model weights
│   ├── sync/
│   │   └── mobile_sync.py                # BulkWriter batch ingestion service
│   ├── main.py                           # Cloud Functions Gen 2 entrypoints
│   └── requirements.txt                  # Python dependencies (NumPy, SciPy, Pillow, Firebase Admin)
├── firestore/
│   ├── firestore.rules                   # Multi-tenant security rules
│   └── firestore.indexes.json            # Composite index definitions
├── storage.rules                         # Cloud Storage security rules
├── firebase.json                         # Firebase configuration & emulator definitions
├── .firebaserc                           # Project alias configuration (`coconut-pathology-detection`)
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites
- **Python**: 3.11+
- **Node.js**: v18.0.0 or higher
- **Firebase CLI**: `npm install -g firebase-tools`
- **Java JRE 17+**: Required for local Firestore emulator

### Local Development Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/R26-SE-016/Coconut-Pathology-Detection-Service.git
   cd Coconut-Pathology-Detection-Service/functions
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the Firebase Emulator Suite**:
   ```bash
   cd ..
   firebase emulators:start --only functions,firestore
   ```
   * Functions will listen on `http://127.0.0.1:5001`.
   * Firestore will listen on `http://127.0.0.1:8080`.
   * Emulator UI is accessible at [http://127.0.0.1:4000](http://127.0.0.1:4000).

---

## 📡 API Request Examples

### 1. Run Aerial Spectral Surveillance (`POST /process_aerial_spectral`)
```http
POST http://127.0.0.1:5001/coconut-pathology-detection/asia-south1/process_aerial_spectral
Content-Type: application/json

{
  "image": "data:image/png;base64,iVBORw0KGgo...",
  "index_type": "VARI",
  "estate_id": "estate_001",
  "gps_bounds": {
    "lat": 7.2906,
    "lng": 80.6337,
    "span_lat": 0.006,
    "span_lng": 0.006
  }
}
```

### 2. Batch Sync Leaf Diagnostics (`POST /sync_mobile_diagnostics`)
```http
POST http://127.0.0.1:5001/coconut-pathology-detection/asia-south1/sync_mobile_diagnostics
Content-Type: application/json

{
  "user_id": "usr_789",
  "device_id": "dev_pixel8",
  "estate_id": "estate_001",
  "batch": [
    {
      "local_id": "scan_001",
      "disease_class": "bud rot",
      "confidence": 0.965,
      "gps": { "lat": 7.2906, "lng": 80.6337 },
      "captured_at": "2026-08-31T08:00:00Z"
    }
  ]
}
```

---

## ☁️ Production Deployment

Deploy directly to Google Cloud in `asia-south1`:

```bash
# Deploy all Cloud Functions
firebase deploy --only functions

# Deploy Firestore security rules
firebase deploy --only firestore:rules

# Deploy Cloud Storage security rules
firebase deploy --only storage:rules
```

---

## 🔬 Research & Citations

Developed under the **SaruPol Research Initiative** (**Project R26-SE-016**). All disease classes, thresholding metrics, and agronomic management directives are formulated in accordance with the **Coconut Research Institute (CRI) of Sri Lanka**.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
