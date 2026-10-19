"""
Local Test Server for Coconut Pathology Detection Service (System A & B)
Enables 100% offline local testing of the aerial spectral pipeline without needing Firebase CLI or Java JRE.
Listens on port 5001 matching SaruPol-Gateway's PATHOLOGY_SERVICE_URL.
"""

import base64
import json
import logging
from flask import Flask, request, jsonify
from inference.spectral_pipeline import SpectralInferencePipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("PathologyLocalServer")

app = Flask(__name__)
pipeline = SpectralInferencePipeline()

@app.route("/health", methods=["GET"])
@app.route("/coconut-pathology-detection/asia-south1/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "online",
        "service": "Coconut-Pathology-Detection-Service",
        "mode": "local_test_server",
        "system_a": "Area-Based Connected Component Spectral Engine"
    }), 200

@app.route("/process_aerial_spectral", methods=["POST"])
@app.route("/coconut-pathology-detection/asia-south1/process_aerial_spectral", methods=["POST"])
def process_spectral():
    try:
        image_bytes = None
        nir_bytes = None
        index_type = "VARI"
        estate_id = "estate_001"
        gps_bounds = None

        if request.content_type and "multipart/form-data" in request.content_type:
            if "image" in request.files:
                image_bytes = request.files["image"].read()
            if "nir_image" in request.files:
                nir_bytes = request.files["nir_image"].read()
            index_type = request.form.get("index_type", "VARI")
            estate_id = request.form.get("estate_id", "estate_001")
            bounds_raw = request.form.get("gps_bounds")
            if bounds_raw:
                try:
                    gps_bounds = json.loads(bounds_raw)
                except Exception:
                    pass
        else:
            body = request.get_json(silent=True) or {}
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
            return jsonify({"error": "Missing required image payload."}), 400

        logger.info(f"Processing aerial orthomosaic | Index: {index_type} | Estate: {estate_id}")
        result = pipeline.process(
            image_bytes=image_bytes,
            index_type=index_type,
            nir_bytes=nir_bytes,
            gps_bounds=gps_bounds,
        )

        res_dict = result.to_dict()
        res_dict["estate_id"] = estate_id

        logger.info(
            f"Analysis complete | Estimated Palms: {result.estimated_palms_count} | "
            f"Hotspots: {len(result.hotspots)} | Grade: {result.estate_health_grade}"
        )
        return jsonify(res_dict), 200

    except Exception as exc:
        logger.error(f"Inference error: {exc}", exc_info=True)
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    print("=" * 70)
    print("  Coconut Pathology Detection Service — Local Test Server")
    print("  Listening on http://127.0.0.1:5001")
    print("  Compatible with SaruPol-Gateway (PORT=8000)")
    print("=" * 70)
    app.run(host="127.0.0.1", port=5001, debug=False)
