# 🥥 Coconut Pathology Detection Service

**Project R26-SE-016** — Multiscale Computer Vision Ecosystem for Coconut Pathology

A serverless Python backend built on **Firebase Cloud Functions (Gen 2)** and **Firestore** that operationalizes two independent deep-learning detection systems for coconut palm diseases.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    CLOUD INFRASTRUCTURE                     │
│                                                             │
│  ┌──────────────────────┐    ┌──────────────────────────┐   │
│  │     SYSTEM A (UAV)   │    │   SYSTEM B (Mobile)      │   │
│  │                      │    │                          │   │
│  │  Cloud Storage       │    │  React Native App        │   │
│  │       │              │    │       │                  │   │
│  │       ▼              │    │       ▼                  │   │
│  │  on_orthomosaic_     │    │  sync_mobile_            │   │
│  │  uploaded()          │    │  diagnostics()           │   │
│  │       │              │    │       │                  │   │
│  │  SAHI Slicing        │    │  BulkWriter              │   │
│  │  1024×1024 tiles     │    │  Batch Writes            │   │
│  │       │              │    │       │                  │   │
│  │  YOLOv11 Inference   │    │       │                  │   │
│  │       │              │    │       │                  │   │
│  │  Cross-Tile NMS      │    │       │                  │   │
│  │       │              │    │       │                  │   │
│  │       ▼              │    │       ▼                  │   │
│  │  Firestore           │    │  Firestore               │   │
│  │  (heatmaps/)         │    │  (diagnostics/)          │   │
│  └──────────────────────┘    └──────────────────────────┘   │
│                                                             │
│           Systems A & B are FULLY INDEPENDENT               │
└─────────────────────────────────────────────────────────────┘
```

### System A — Macroscopic Detection (UAV)
> **For**: Plantation owners with drone capability

- **Input**: 4K UAV orthomosaics uploaded to Cloud Storage
- **Model**: YOLOv11 (server-side, bundled weights)
- **Pipeline**: SAHI slicing → 1024×1024 tiles (20% overlap) → YOLOv11 → Cross-tile NMS
- **Output**: Pathological heatmaps in `heatmaps/` collection
- **Detects**: V-cuts, scorching, wilting

### System B — Microscopic Detection (Mobile)
> **For**: Small coconut farmers using mobile devices

- **Input**: On-device MobileNetV2-INT8 classification results
- **Pipeline**: React Native app → HTTP POST → BulkWriter → Firestore
- **Output**: Georeferenced diagnostics in `diagnostics/` collection
- **Syncs**: Disease class, confidence, GPS coordinates

---

## Cloud Functions

| Function | Trigger | System | Purpose |
|---|---|---|---|
| `on_orthomosaic_uploaded` | Storage event | A | Process UAV orthomosaic → heatmap |
| `get_estate_heatmap` | HTTP GET | A | Fetch heatmaps for an estate |
| `sync_mobile_diagnostics` | HTTP POST | B | Batch-write mobile diagnostic results |
| `get_diagnostic_history` | HTTP GET | B | Fetch user's diagnostic history |

---

## Project Structure

```
├── functions/
│   ├── main.py                   # Cloud Functions entry points
│   ├── requirements.txt          # Python dependencies
│   ├── inference/
│   │   ├── sahi_pipeline.py      # SAHI + YOLOv11 inference
│   │   └── nms.py                # Cross-tile NMS merging
│   ├── sync/
│   │   └── mobile_sync.py        # Mobile batch sync service
│   └── models/
│       └── coconut_yolov11.pt    # Bundled model weights (gitignored)
├── firestore/
│   ├── schema.json               # Firestore data schema
│   └── firestore.rules           # Multi-tenant security rules
└── README.md
```

---

## Firestore Collections

| Collection | System | Description |
|---|---|---|
| `users/` | Shared | User profiles with roles & estate assignments |
| `estates/` | Shared | Estate metadata with geographic bounds |
| `diagnostics/` | A & B | Georeferenced results (`source` field distinguishes origin) |
| `heatmaps/` | A only | NMS-merged detection heatmaps from UAV pipeline |
| `knowledge_base/` | Reserved | CRI expert data (managed by Advisory System team) |

---

## Setup & Deployment

### Prerequisites
- Firebase project with **Firestore (Native mode)** and **Cloud Storage**
- Firebase CLI installed (`npm install -g firebase-tools`)
- Python 3.11+

### Deploy

```bash
# Login to Firebase
firebase login

# Deploy Cloud Functions
firebase deploy --only functions

# Deploy Firestore rules
firebase deploy --only firestore:rules
```

### Local Testing

```bash
# Start Firebase Emulators
firebase emulators:start

# Test System B sync endpoint
curl -X POST http://localhost:5001/<PROJECT_ID>/asia-south1/sync_mobile_diagnostics \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "test_user_001",
    "device_id": "device_abc",
    "estate_id": "estate_001",
    "batch": [{
      "disease_class": "WCLWD",
      "confidence": 0.92,
      "gps": { "lat": 7.2906, "lng": 80.6337 },
      "captured_at": "2026-05-09T10:30:00Z"
    }]
  }'
```

### Model Training & Notebooks

- **System B (Mobile)**: [Coconut Pathology Training Notebook](file:///f:/GitHub/Research/Coconut-Pathology-Detection-Service/notebooks/coconut_pathology_training.ipynb)
  - Targets MobileNetV2 with INT8 quantization for <35ms latency.
  - **Dataset V2**: Now includes 6 classes including a **Healthy** baseline.
  - Optimized for React Native on-device inference.

### Model Weights

## Security Model

Three-tier role-based access:
- **Field Officers** — Read/write own diagnostics within their estate
- **Managers** — Read all diagnostics and heatmaps within their region (cross-estate hotspot visibility)
- **Admins** — Full read access across all estates

> ⚠️ Authentication module is not yet integrated. Security rules use placeholder `request.auth.token` custom claims.

---

## API Reference

### POST `/sync_mobile_diagnostics`

**Request:**
```json
{
  "user_id": "uid_abc123",
  "device_id": "device_xyz",
  "estate_id": "estate_001",
  "batch": [
    {
      "disease_class": "WCLWD",
      "confidence": 0.92,
      "gps": { "lat": 7.2906, "lng": 80.6337 },
      "captured_at": "2026-05-09T10:30:00Z",
      "image_ref": "mobile_uploads/uid_abc123/img_001.jpg",
      "local_id": "local-uuid-001"
    }
  ]
}
```

**Response:**
```json
{
  "synced_count": 1,
  "failed_ids": [],
  "server_timestamp": "2026-05-09T18:30:00+00:00"
}
```

### GET `/get_estate_heatmap?estate_id=estate_001&limit=10`

### GET `/get_diagnostic_history?user_id=uid_abc123&limit=50`

---

## Team

| Member | Component |
|---|---|
| Lakshan H.G.J.S. | Detection Pipelines & Backend (this repository) |
| [Teammate] | Advisory System & Knowledge Base |
