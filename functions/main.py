# ══════════════════════════════════════════════════════════════════════
# Coconut Pathology Detection Service — Cloud Functions (Gen 2)
# Project: R26-SE-016 — Multiscale Computer Vision Ecosystem
#
# This module exposes four Firebase Cloud Functions:
#
#   System A (UAV / Macroscopic):
#     • on_orthomosaic_uploaded  — Storage trigger → SAHI + YOLOv11
#     • get_estate_heatmap       — HTTP GET → Fetch heatmap data
#
#   System B (Mobile / Microscopic):
#     • sync_mobile_diagnostics  — HTTP POST → Batch-write diagnostics
#     • get_diagnostic_history   — HTTP GET → Fetch user diagnostics
#
# Systems A and B are COMPLETELY INDEPENDENT.
# ══════════════════════════════════════════════════════════════════════

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone

import firebase_admin
from firebase_admin import firestore as admin_firestore
from firebase_admin import storage as admin_storage
from firebase_functions import https_fn, logger, options, storage_fn
from google.cloud.firestore import Client as FirestoreClient

# ── Firebase Initialisation ──────────────────────────────────────────
# Initialise once per cold start; reused across all function invocations.

firebase_admin.initialize_app()


def _get_db() -> FirestoreClient:
    """Return the Firestore client (lazy, cached by the Admin SDK)."""
    return admin_firestore.client()


# ── Global Options ───────────────────────────────────────────────────

options.set_global_options(region="asia-south1")


# ══════════════════════════════════════════════════════════════════════
#  SYSTEM A — MACROSCOPIC INFERENCE PIPELINE (UAV + YOLOv11)
# ══════════════════════════════════════════════════════════════════════


@storage_fn.on_object_finalized(
    memory=options.MemoryOption.GB_4,
    timeout_sec=540,
    cpu=2,
)
def on_orthomosaic_uploaded(
    event: storage_fn.CloudEvent[storage_fn.StorageObjectData],
) -> None:
    """
    Triggered when a new UAV orthomosaic is uploaded to Cloud Storage.

    Expected path convention:
        ``orthomosaics/{estateId}/{filename}.tif``

    Pipeline:
        1. Download image to /tmp
        2. SAHI slicing (1024×1024, 20 % overlap)
        3. YOLOv11 inference per tile
        4. Cross-tile NMS merging
        5. Write heatmap document to Firestore
        6. Cleanup /tmp
    """
    file_path: str = event.data.name
    bucket_name: str = event.data.bucket

    # ── Guard: only process files in orthomosaics/ ────────────────
    if not file_path.startswith("orthomosaics/"):
        logger.info(f"Ignoring non-orthomosaic upload: {file_path}")
        return

    # Parse estate ID from the path
    path_parts = file_path.split("/")
    if len(path_parts) < 3:
        logger.error(
            f"Invalid path structure: {file_path}. "
            f"Expected orthomosaics/{{estateId}}/{{filename}}"
        )
        return

    estate_id = path_parts[1]
    filename = path_parts[-1]

    logger.info(
        f"[System A] Processing orthomosaic — "
        f"estate={estate_id}, file={filename}, bucket={bucket_name}"
    )

    # ── Step 1: Download to /tmp ─────────────────────────────────
    local_path = os.path.join(tempfile.gettempdir(), filename)

    try:
        bucket = admin_storage.bucket(bucket_name)
        blob = bucket.blob(file_path)
        blob.download_to_filename(local_path)
        logger.info(f"Downloaded {file_path} → {local_path}")
    except Exception as exc:
        logger.error(f"Failed to download {file_path}: {exc}")
        return

    try:
        # ── Step 2 & 3: SAHI + YOLOv11 inference ────────────────
        from inference.sahi_pipeline import SahiInferencePipeline

        pipeline = SahiInferencePipeline()
        pipeline_output = pipeline.run(local_path)

        logger.info(
            f"Inference complete — {len(pipeline_output.detections)} "
            f"raw detections from {pipeline_output.num_slices} tiles"
        )

        # ── Step 4: Cross-tile NMS ───────────────────────────────
        from inference.nms import CrossTileNMS

        nms = CrossTileNMS()
        heatmap = nms.merge(pipeline_output.detections)

        # ── Step 5: Write to Firestore ───────────────────────────
        db = _get_db()
        now = datetime.now(timezone.utc)

        heatmap_doc = {
            "estate_id": estate_id,
            "image_ref": f"gs://{bucket_name}/{file_path}",
            "image_dimensions": {
                "width": pipeline_output.image_width,
                "height": pipeline_output.image_height,
            },
            "detections": [
                {
                    "bbox": det.bbox,
                    "class": det.category_name,
                    "confidence": det.confidence,
                    "category_id": det.category_id,
                }
                for det in heatmap.detections
            ],
            "summary": {
                "total": heatmap.total_detections,
                "by_class": {
                    cls_name: {
                        "count": stats.count,
                        "mean_confidence": stats.mean_confidence,
                        "max_confidence": stats.max_confidence,
                    }
                    for cls_name, stats in heatmap.summary.items()
                },
            },
            "created_at": now.isoformat(),
            "processed_at": admin_firestore.SERVER_TIMESTAMP,
            "processed_by": "on_orthomosaic_uploaded/v1",
        }

        doc_ref = db.collection("heatmaps").document()
        doc_ref.set(heatmap_doc)

        logger.info(
            f"[System A] Heatmap written → heatmaps/{doc_ref.id} "
            f"({heatmap.total_detections} detections for estate {estate_id})"
        )

    except Exception as exc:
        logger.error(f"[System A] Pipeline failed for {file_path}: {exc}")
        raise

    finally:
        # ── Step 6: Cleanup ──────────────────────────────────────
        if os.path.exists(local_path):
            os.remove(local_path)
            logger.info(f"Cleaned up {local_path}")


# ── System A: HTTP endpoint to retrieve heatmap data ─────────────────

@https_fn.on_request(
    cors=options.CorsOptions(cors_origins="*", cors_methods=["GET"]),
    memory=options.MemoryOption.MB_256,
    timeout_sec=30,
)
def get_estate_heatmap(req: https_fn.Request) -> https_fn.Response:
    """
    Fetch heatmap data for a specific estate.

    Query params:
        - estate_id (required): The estate identifier
        - limit (optional): Max number of heatmaps to return (default 10)

    Returns:
        JSON array of heatmap documents, newest first.
    """
    if req.method != "GET":
        return https_fn.Response(
            json.dumps({"error": "Method not allowed. Use GET."}),
            status=405,
            content_type="application/json",
        )

    estate_id = req.args.get("estate_id")
    if not estate_id:
        return https_fn.Response(
            json.dumps({"error": "Missing required query param: 'estate_id'"}),
            status=400,
            content_type="application/json",
        )

    limit = min(int(req.args.get("limit", 10)), 50)

    try:
        db = _get_db()
        query = (
            db.collection("heatmaps")
            .where("estate_id", "==", estate_id)
            .order_by("created_at", direction="DESCENDING")
            .limit(limit)
        )

        docs = query.stream()
        results = []
        for doc in docs:
            data = doc.to_dict()
            data["id"] = doc.id
            results.append(data)

        logger.info(
            f"[System A] Returned {len(results)} heatmaps for estate={estate_id}"
        )

        return https_fn.Response(
            json.dumps({"estate_id": estate_id, "heatmaps": results}, default=str),
            status=200,
            content_type="application/json",
        )

    except Exception as exc:
        logger.error(f"Failed to fetch heatmaps: {exc}")
        return https_fn.Response(
            json.dumps({"error": "Internal server error"}),
            status=500,
            content_type="application/json",
        )


# ══════════════════════════════════════════════════════════════════════
#  SYSTEM B — OFFLINE-FIRST MOBILE SYNC (MobileNetV2 Results)
# ══════════════════════════════════════════════════════════════════════


@https_fn.on_request(
    cors=options.CorsOptions(cors_origins="*", cors_methods=["POST"]),
    memory=options.MemoryOption.MB_512,
    timeout_sec=120,
)
def sync_mobile_diagnostics(req: https_fn.Request) -> https_fn.Response:
    """
    HTTP endpoint for the React Native app to sync diagnostic results.

    Accepts a batch of on-device MobileNetV2-INT8 classification results
    and writes them to the ``diagnostics`` collection using BulkWriter.

    Request body (JSON)::

        {
            "user_id":   "uid_abc123",
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

    Response (JSON)::

        {
            "synced_count": 5,
            "failed_ids": [],
            "server_timestamp": "2026-05-09T18:30:00+00:00"
        }
    """
    if req.method != "POST":
        return https_fn.Response(
            json.dumps({"error": "Method not allowed. Use POST."}),
            status=405,
            content_type="application/json",
        )

    # Parse JSON body
    try:
        body = req.get_json(silent=True)
        if body is None:
            raise ValueError("Empty or invalid JSON body.")
    except Exception as exc:
        return https_fn.Response(
            json.dumps({"error": f"Invalid request body: {exc}"}),
            status=400,
            content_type="application/json",
        )

    # Validate payload
    from sync.mobile_sync import MobileSyncService, ValidationError, validate_sync_request

    try:
        sync_request = validate_sync_request(body)
    except ValidationError as exc:
        return https_fn.Response(
            json.dumps({"error": str(exc)}),
            status=422,
            content_type="application/json",
        )

    # Execute batch write
    try:
        db = _get_db()
        service = MobileSyncService(db)
        receipt = service.sync(sync_request)

        logger.info(
            f"[System B] Sync complete — "
            f"user={sync_request.user_id}, synced={receipt.synced_count}"
        )

        return https_fn.Response(
            json.dumps({
                "synced_count": receipt.synced_count,
                "failed_ids": receipt.failed_ids,
                "server_timestamp": receipt.server_timestamp,
            }),
            status=200,
            content_type="application/json",
        )

    except Exception as exc:
        logger.error(f"[System B] Sync failed: {exc}")
        return https_fn.Response(
            json.dumps({"error": "Internal server error during sync."}),
            status=500,
            content_type="application/json",
        )


# ── System B: HTTP endpoint to retrieve user diagnostic history ──────

@https_fn.on_request(
    cors=options.CorsOptions(cors_origins="*", cors_methods=["GET"]),
    memory=options.MemoryOption.MB_256,
    timeout_sec=30,
)
def get_diagnostic_history(req: https_fn.Request) -> https_fn.Response:
    """
    Fetch diagnostic history for a specific user.

    Query params:
        - user_id (required): The user identifier
        - estate_id (optional): Filter by estate
        - limit (optional): Max results (default 50, max 200)

    Returns:
        JSON array of diagnostic documents, newest first.
    """
    if req.method != "GET":
        return https_fn.Response(
            json.dumps({"error": "Method not allowed. Use GET."}),
            status=405,
            content_type="application/json",
        )

    user_id = req.args.get("user_id")
    if not user_id:
        return https_fn.Response(
            json.dumps({"error": "Missing required query param: 'user_id'"}),
            status=400,
            content_type="application/json",
        )

    estate_id = req.args.get("estate_id")
    limit = min(int(req.args.get("limit", 50)), 200)

    try:
        db = _get_db()
        query = db.collection("diagnostics").where("user_id", "==", user_id)

        if estate_id:
            query = query.where("estate_id", "==", estate_id)

        query = query.order_by("created_at", direction="DESCENDING").limit(limit)

        docs = query.stream()
        results = []
        for doc in docs:
            data = doc.to_dict()
            data["id"] = doc.id
            # Convert GeoPoint to serializable dict
            if "location" in data and hasattr(data["location"], "latitude"):
                data["location"] = {
                    "lat": data["location"].latitude,
                    "lng": data["location"].longitude,
                }
            results.append(data)

        logger.info(
            f"[System B] Returned {len(results)} diagnostics for user={user_id}"
        )

        return https_fn.Response(
            json.dumps({
                "user_id": user_id,
                "count": len(results),
                "diagnostics": results,
            }, default=str),
            status=200,
            content_type="application/json",
        )

    except Exception as exc:
        logger.error(f"Failed to fetch diagnostics: {exc}")
        return https_fn.Response(
            json.dumps({"error": "Internal server error"}),
            status=500,
            content_type="application/json",
        )
