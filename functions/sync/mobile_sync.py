# ──────────────────────────────────────────────────────────────────────
# System B — Offline-First Mobile Diagnostic Sync Service
#
# Accepts batches of on-device MobileNetV2-INT8 classification results
# from the React Native app and writes them to Firestore using
# BulkWriter for high-throughput performance (up to 5,000 daily users).
# ──────────────────────────────────────────────────────────────────────

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from firebase_admin import firestore as admin_firestore
from firebase_functions import logger
from google.cloud.firestore import Client as FirestoreClient
from google.cloud.firestore_v1.base_document import DocumentReference


# ── Constants ────────────────────────────────────────────────────────

MAX_BATCH_SIZE = 500          # Firestore batch limit
DIAGNOSTICS_COLLECTION = "diagnostics"


# ── DTOs ─────────────────────────────────────────────────────────────

@dataclass
class DiagnosticItem:
    """A single classification result from the on-device MobileNetV2."""

    disease_class: str           # e.g. "WCLWD", "Bud_Rot", "Healthy"
    confidence: float            # 0.0 – 1.0
    latitude: float
    longitude: float
    captured_at: str             # ISO 8601 timestamp from the device
    image_ref: Optional[str] = None   # Optional GCS path if the image was uploaded
    local_id: Optional[str] = None    # Client-side UUID for dedup / ack


@dataclass
class SyncRequest:
    """Payload sent by the React Native app."""

    user_id: str
    device_id: str
    estate_id: str
    batch: List[DiagnosticItem] = field(default_factory=list)


@dataclass
class SyncReceipt:
    """Response returned to the mobile client."""

    synced_count: int = 0
    failed_ids: List[str] = field(default_factory=list)
    server_timestamp: str = ""


# ── Validation ───────────────────────────────────────────────────────

class ValidationError(Exception):
    """Raised when the incoming sync payload fails validation."""


def validate_sync_request(data: Dict[str, Any]) -> SyncRequest:
    """
    Parse and validate the raw JSON body into a ``SyncRequest``.

    Raises ``ValidationError`` with a descriptive message on failure.
    """
    # Required top-level fields
    for key in ("user_id", "device_id", "estate_id", "batch"):
        if key not in data:
            raise ValidationError(f"Missing required field: '{key}'")

    batch_raw = data["batch"]
    if not isinstance(batch_raw, list):
        raise ValidationError("'batch' must be an array.")

    if len(batch_raw) == 0:
        raise ValidationError("'batch' must contain at least one item.")

    if len(batch_raw) > MAX_BATCH_SIZE:
        raise ValidationError(
            f"Batch size {len(batch_raw)} exceeds the maximum of "
            f"{MAX_BATCH_SIZE}. Split into multiple requests."
        )

    items: List[DiagnosticItem] = []
    for i, item in enumerate(batch_raw):
        # Validate each diagnostic item
        for req_field in ("disease_class", "confidence", "gps", "captured_at"):
            if req_field not in item:
                raise ValidationError(
                    f"Batch item [{i}] missing required field: '{req_field}'"
                )

        gps = item["gps"]
        if not isinstance(gps, dict) or "lat" not in gps or "lng" not in gps:
            raise ValidationError(
                f"Batch item [{i}] 'gps' must be {{ lat, lng }}."
            )

        confidence = float(item["confidence"])
        if not (0.0 <= confidence <= 1.0):
            raise ValidationError(
                f"Batch item [{i}] confidence {confidence} out of range [0, 1]."
            )

        items.append(
            DiagnosticItem(
                disease_class=str(item["disease_class"]),
                confidence=confidence,
                latitude=float(gps["lat"]),
                longitude=float(gps["lng"]),
                captured_at=str(item["captured_at"]),
                image_ref=item.get("image_ref"),
                local_id=item.get("local_id", str(uuid.uuid4())),
            )
        )

    return SyncRequest(
        user_id=str(data["user_id"]),
        device_id=str(data["device_id"]),
        estate_id=str(data["estate_id"]),
        batch=items,
    )


# ── Service ──────────────────────────────────────────────────────────

class MobileSyncService:
    """
    High-throughput batch-write service for mobile diagnostic results.

    Uses Firestore ``BulkWriter`` to parallelise writes across threads,
    achieving significantly better throughput than sequential ``set()``
    calls or the 500-op ``WriteBatch``.

    Usage::

        service = MobileSyncService(db)
        receipt = service.sync(validated_request)
    """

    def __init__(self, db: FirestoreClient) -> None:
        self.db = db

    def sync(self, request: SyncRequest) -> SyncReceipt:
        """
        Write a batch of mobile diagnostics to Firestore.

        Returns a ``SyncReceipt`` acknowledging which items succeeded.
        """
        now = datetime.now(timezone.utc)
        synced = 0
        failed_ids: List[str] = []

        # Use BulkWriter for parallelised, high-throughput writes
        bulk_writer = self.db.bulk_writer()

        for item in request.batch:
            try:
                doc_ref: DocumentReference = self.db.collection(
                    DIAGNOSTICS_COLLECTION
                ).document()

                doc_data = {
                    "user_id": request.user_id,
                    "device_id": request.device_id,
                    "estate_id": request.estate_id,
                    "disease_class": item.disease_class,
                    "confidence": item.confidence,
                    "location": admin_firestore.firestore.GeoPoint(
                        item.latitude, item.longitude
                    ),
                    "source": "mobile_v2",
                    "image_ref": item.image_ref,
                    "local_id": item.local_id,
                    "captured_at": item.captured_at,
                    "synced_at": now.isoformat(),
                    "created_at": admin_firestore.SERVER_TIMESTAMP,
                }

                bulk_writer.set(doc_ref, doc_data)
                synced += 1

            except Exception as exc:
                logger.error(
                    f"Failed to queue diagnostic item "
                    f"(local_id={item.local_id}): {exc}"
                )
                failed_ids.append(item.local_id or "unknown")

        # Flush all queued writes — blocks until complete
        try:
            bulk_writer.flush()
            bulk_writer.close()
        except Exception as exc:
            logger.error(f"BulkWriter flush failed: {exc}")
            # If flush fails, all items in this batch are considered failed
            return SyncReceipt(
                synced_count=0,
                failed_ids=[
                    item.local_id or "unknown" for item in request.batch
                ],
                server_timestamp=now.isoformat(),
            )

        logger.info(
            f"Mobile sync complete — {synced} written, "
            f"{len(failed_ids)} failed for user={request.user_id}"
        )

        return SyncReceipt(
            synced_count=synced,
            failed_ids=failed_ids,
            server_timestamp=now.isoformat(),
        )
