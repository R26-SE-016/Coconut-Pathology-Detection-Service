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
from google.cloud.firestore import DocumentReference

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
    bucket="coconut-pathology-detection.appspot.com",
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
        # ── Step 2: Spectral & Morphological Inference ───────────
        from inference.spectral_pipeline import SpectralInferencePipeline

        with open(local_path, "rb") as f:
            img_bytes = f.read()

        pipeline = SpectralInferencePipeline()
        result = pipeline.process(image_bytes=img_bytes, index_type="VARI")
        res_dict = result.to_dict()

        logger.info(
            f"[System A] Orthomosaic analyzed — "
            f"{result.estimated_palms_count} palms detected, "
            f"{len(result.hotspots)} hotspots flagged"
        )

        # ── Step 3: Write to Firestore ───────────────────────────
        db = _get_db()
        now = datetime.now(timezone.utc)

        heatmap_doc = {
            "estate_id": estate_id,
            "image_ref": f"gs://{bucket_name}/{file_path}",
            "image_dimensions": res_dict["image_dimensions"],
            "statistics": res_dict["statistics"],
            "hotspots": [h.to_dict() for h in result.hotspots],
            "created_at": now.isoformat(),
            "processed_at": admin_firestore.SERVER_TIMESTAMP,
            "processed_by": "on_orthomosaic_uploaded/v1",
            "source": "storage_orthomosaic_trigger",
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
    cors=options.CorsOptions(cors_origins="*", cors_methods=["GET", "OPTIONS"]),
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
    cors=options.CorsOptions(cors_origins="*", cors_methods=["POST", "OPTIONS"]),
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
    cors=options.CorsOptions(cors_origins="*", cors_methods=["GET", "OPTIONS"]),
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

# ── System B: HTTP endpoint to run real-time inference on the backend ─────────

@https_fn.on_request(
    cors=options.CorsOptions(cors_origins="*", cors_methods=["POST", "OPTIONS"]),
    memory=options.MemoryOption.MB_512,
    timeout_sec=120,
)
def predict_mobile_disease(req: https_fn.Request) -> https_fn.Response:
    """
    Run TFLite MobileNetV2 inference on the backend.
    Accepts multipart/form-data with an 'image' file.
    """
    if req.method != "POST":
        return https_fn.Response(
            json.dumps({"error": "Method not allowed. Use POST."}),
            status=405,
            content_type="application/json",
        )

    image_file = req.files.get("image")
    if not image_file:
        return https_fn.Response(
            json.dumps({"error": "Missing 'image' in multipart/form-data"}),
            status=400,
            content_type="application/json",
        )

    try:
        import time
        import numpy as np
        from PIL import Image
        import io
        import tensorflow as tf

        start_time = time.time()

        # Load and preprocess image
        img_bytes = image_file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img = img.resize((224, 224), Image.Resampling.NEAREST)
        
        # Convert to numpy and match the TFLite model's expected UINT8 input
        input_data = np.expand_dims(np.array(img, dtype=np.uint8), axis=0)

        # Load TFLite model
        model_path = os.path.join(os.path.dirname(__file__), "models", "system_b", "system_b_baseline_int8.tflite")
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Run inference
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        probs = interpreter.get_tensor(output_details[0]['index'])[0]
        
        inference_time_ms = int((time.time() - start_time) * 1000)

        # Post-process
        CLASS_NAMES = [
            'bud root dropping',
            'bud rot',
            'gray leaf spot',
            'healthy leaves',
            'leaf rot',
            'stembleeding',
        ]
        
        # INT8 outputs are usually UINT8 (0-255). Convert to 0-1 probability.
        # We check the dtype of the output tensor to decide how to process.
        output_dtype = output_details[0]['dtype']
        
        def to_prob(v):
            if output_dtype == np.uint8 or output_dtype == np.int8:
                # Quantized: map 0-255 to 0-1 (simple approximation for Softmax)
                return float(v) / 255.0
            return float(v)

        max_idx = int(np.argmax(probs))
        top_confidence = to_prob(probs[max_idx])
        
        all_predictions = [
            {"class": cls_name, "confidence": to_prob(prob)}
            for cls_name, prob in zip(CLASS_NAMES, probs)
        ]
        all_predictions.sort(key=lambda x: x["confidence"], reverse=True)

        return https_fn.Response(
            json.dumps({
                "disease_class": CLASS_NAMES[max_idx],
                "confidence": top_confidence,
                "all_predictions": all_predictions,
                "inference_time_ms": inference_time_ms
            }),
            status=200,
            content_type="application/json",
        )

    except Exception as exc:
        logger.error(f"Inference failed: {exc}")
        return https_fn.Response(
            json.dumps({"error": str(exc)}),
            status=500,
            content_type="application/json",
        )


# ══════════════════════════════════════════════════════════════════════
#  SYSTEM A — DUAL-INDEX SPECTRAL ANALYSIS (NDVI & VARI + HOTSPOTS)
# ══════════════════════════════════════════════════════════════════════


@https_fn.on_request(
    cors=options.CorsOptions(cors_origins="*", cors_methods=["POST", "OPTIONS"]),
    memory=options.MemoryOption.GB_1,
    timeout_sec=180,
)
def process_aerial_spectral(req: https_fn.Request) -> https_fn.Response:
    """
    Process aerial drone imagery to calculate NDVI or VARI spectral indices,
    segment canopy health distributions, and detect stressed hotspots.

    Accepts multipart/form-data or JSON with base64 image:
        - image (file or base64 string): Primary aerial image
        - nir_image (optional file/base64): Companion NIR image for NDVI
        - index_type (str): 'NDVI' or 'VARI' (default: 'VARI')
        - estate_id (str): Associated estate identifier
        - gps_bounds (optional JSON string/dict): { lat, lng, span_lat, span_lng }
    """
    if req.method != "POST":
        return https_fn.Response(
            json.dumps({"error": "Method not allowed. Use POST."}),
            status=405,
            content_type="application/json",
        )

    try:
        import base64
        from inference.spectral_pipeline import SpectralInferencePipeline

        image_bytes: Optional[bytes] = None
        nir_bytes: Optional[bytes] = None
        index_type = "VARI"
        estate_id = "default_estate"
        gps_bounds = None

        # Check content type: multipart vs json
        if req.content_type and "multipart/form-data" in req.content_type:
            image_file = req.files.get("image")
            if image_file:
                image_bytes = image_file.read()

            nir_file = req.files.get("nir_image")
            if nir_file:
                nir_bytes = nir_file.read()

            index_type = req.form.get("index_type", "VARI")
            estate_id = req.form.get("estate_id", "estate_001")
            bounds_raw = req.form.get("gps_bounds")
            if bounds_raw:
                try:
                    gps_bounds = json.loads(bounds_raw)
                except Exception:
                    pass
        else:
            body = req.get_json(silent=True) or {}
            img_b64 = body.get("image") or body.get("imageBase64")
            if img_b64:
                if "," in img_b64:
                    img_b64 = img_b64.split(",", 1)[1]
                image_bytes = base64.b64decode(img_b64)

            nir_b64 = body.get("nir_image") or body.get("nirBase64")
            if nir_b64:
                if "," in nir_b64:
                    nir_b64 = nir_b64.split(",", 1)[1]
                nir_bytes = base64.b64decode(nir_b64)

            index_type = body.get("index_type", "VARI")
            estate_id = body.get("estate_id", "estate_001")
            gps_bounds = body.get("gps_bounds")

        if not image_bytes:
            return https_fn.Response(
                json.dumps({"error": "Missing required image file/payload."}),
                status=400,
                content_type="application/json",
            )

        # Run pipeline
        pipeline = SpectralInferencePipeline()
        result = pipeline.process(
            image_bytes=image_bytes,
            index_type=index_type,
            nir_bytes=nir_bytes,
            gps_bounds=gps_bounds,
        )

        res_dict = result.to_dict()
        res_dict["estate_id"] = estate_id
        res_dict["created_at"] = datetime.now(timezone.utc).isoformat()

        # Save record to Firestore
        try:
            db = _get_db()
            batch = db.batch()

            # Save heatmap record
            heatmap_ref = db.collection("heatmaps").document()
            heatmap_data = {
                "estate_id": estate_id,
                "index_type": result.index_type,
                "image_dimensions": res_dict["image_dimensions"],
                "statistics": res_dict["statistics"],
                "created_at": res_dict["created_at"],
                "source": "aerial_spectral",
            }
            batch.set(heatmap_ref, heatmap_data)

            # Save each individual hotspot for field mobile inspection
            for hs in result.hotspots:
                hs_ref = db.collection("canopy_hotspots").document(hs.id)
                hs_data = hs.to_dict()
                hs_data["estate_id"] = estate_id
                hs_data["heatmap_id"] = heatmap_ref.id
                hs_data["index_type"] = result.index_type
                hs_data["created_at"] = res_dict["created_at"]
                batch.set(hs_ref, hs_data)

            batch.commit()
            res_dict["heatmap_id"] = heatmap_ref.id
            logger.info(
                f"[System A] Spectral analysis saved: heatmap={heatmap_ref.id}, "
                f"hotspots={len(result.hotspots)} for estate={estate_id}"
            )
        except Exception as db_exc:
            logger.warn(f"[System A] Firestore write skipped/failed: {db_exc}")

        def _numpy_safe(obj):
            if hasattr(obj, "item"):
                return obj.item()
            return str(obj)

        return https_fn.Response(
            json.dumps(res_dict, default=_numpy_safe),
            status=200,
            content_type="application/json",
        )

    except Exception as exc:
        logger.error(f"[System A] Spectral processing failed: {exc}")
        return https_fn.Response(
            json.dumps({"error": str(exc)}),
            status=500,
            content_type="application/json",
        )


@https_fn.on_request(
    cors=options.CorsOptions(cors_origins="*", cors_methods=["GET", "OPTIONS"]),
    memory=options.MemoryOption.MB_256,
    timeout_sec=30,
)
def get_canopy_hotspots(req: https_fn.Request) -> https_fn.Response:
    """
    Fetch active canopy stress hotspots for an estate so mobile field officers
    can perform targeted on-ground leaf inspection.

    Query params:
        - estate_id (required): The estate identifier
        - status (optional): Filter by 'pending', 'inspected', 'resolved' (default: all)
        - limit (optional): Max hotspots (default: 50)
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

    status_filter = req.args.get("status")
    limit = min(int(req.args.get("limit", 50)), 100)

    try:
        db = _get_db()
        query = db.collection("canopy_hotspots").where("estate_id", "==", estate_id)

        if status_filter:
            query = query.where("status", "==", status_filter)

        docs = query.order_by("created_at", direction="DESCENDING").limit(limit).stream()
        results = []
        for doc in docs:
            d = doc.to_dict()
            d["id"] = doc.id
            results.append(d)

        return https_fn.Response(
            json.dumps({"estate_id": estate_id, "count": len(results), "hotspots": results}),
            status=200,
            content_type="application/json",
        )
    except Exception as exc:
        logger.error(f"Failed to fetch canopy hotspots: {exc}")
        return https_fn.Response(
            json.dumps({"error": "Internal server error"}),
            status=500,
            content_type="application/json",
        )


@https_fn.on_request(
    cors=options.CorsOptions(cors_origins="*", cors_methods=["PATCH", "POST", "OPTIONS"]),
    memory=options.MemoryOption.MB_256,
    timeout_sec=30,
)
def update_hotspot_status(req: https_fn.Request) -> https_fn.Response:
    """
    Update inspection status of a canopy hotspot once a field officer diagnoses it.
    """
    if req.method not in ["PATCH", "POST"]:
        return https_fn.Response(
            json.dumps({"error": "Method not allowed. Use PATCH or POST."}),
            status=405,
            content_type="application/json",
        )

    body = req.get_json(silent=True) or {}
    hotspot_id = body.get("hotspot_id") or req.args.get("hotspot_id")
    status_val = body.get("status", "inspected")
    leaf_diag_id = body.get("leaf_diagnostic_id")

    if not hotspot_id:
        return https_fn.Response(
            json.dumps({"error": "Missing 'hotspot_id'"}),
            status=400,
            content_type="application/json",
        )

    try:
        db = _get_db()
        doc_ref = db.collection("canopy_hotspots").document(hotspot_id)
        update_data = {
            "status": status_val,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        if leaf_diag_id:
            update_data["leaf_diagnostic_id"] = leaf_diag_id

        doc_ref.update(update_data)
        return https_fn.Response(
            json.dumps({"success": True, "hotspot_id": hotspot_id, "status": status_val}),
            status=200,
            content_type="application/json",
        )
    except Exception as exc:
        logger.error(f"Failed to update hotspot: {exc}")
        return https_fn.Response(
            json.dumps({"error": str(exc)}),
            status=500,
            content_type="application/json",
        )

