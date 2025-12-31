import os
import sys
import uuid
import logging
from flask import Flask, request, jsonify, send_from_directory
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

current_dir = Path(__file__).parent
project_root = current_dir.parent if current_dir.name == "frontend" else current_dir
sys.path.insert(0, str(project_root))

app = Flask(__name__, static_folder=str(project_root / "frontend"), static_url_path="")

UPLOAD_FOLDER = project_root / "uploads"
GENERATED_FOLDER = project_root / "generated"
HEATMAP_FOLDER = project_root / "heatmaps"

UPLOAD_FOLDER.mkdir(exist_ok=True)
GENERATED_FOLDER.mkdir(exist_ok=True)
HEATMAP_FOLDER.mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ContentGuardAI")


def get_recommendation(prob_fake, prob_real):
    """
    Génère la recommandation selon les probabilités
    """
    if prob_fake >= 85:
        return "REJECT - Strong evidence of manipulation"
    elif prob_fake >= 70:
        return "REJECT - Manual verification recommended"
    elif prob_fake >= 50:
        return "REVIEW - Additional verification recommended"
    elif prob_real >= 85:
        return "ACCEPT - Minimal risk detected"
    elif prob_real >= 70:
        return "ACCEPT - Low risk detected"
    else:
        return "REVIEW - Additional analysis recommended"



try:
    from moderator.moderation_engine import engine as moderation_engine

    MODERATION_LOADED = True
    logger.info(" Moderation loaded")
except Exception as e:
    MODERATION_LOADED = False
    logger.warning(f" Moderation not loaded: {e}")

try:
    from generation.generator_engine import generator as generation_engine

    GENERATION_LOADED = True
    logger.info(" Generation loaded")
except Exception as e:
    GENERATION_LOADED = False
    logger.warning(f" Generation not loaded: {e}")



@app.route("/")
def index():
    return send_from_directory(project_root / "frontend", "index.html")


@app.route("/generate.html")
def generate_page():
    return send_from_directory(project_root / "frontend", "generate.html")


@app.route("/upload/batch", methods=["POST"])
def upload_batch():
    if not MODERATION_LOADED:
        return jsonify({"error": "Moderation unavailable"}), 503

    if "files" not in request.files:
        return jsonify({"error": "No files sent"}), 400

    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "Empty file list"}), 400

    saved_paths = []
    filenames = []

    for file in files:
        if file.filename:
            filename = f"{uuid.uuid4()}_{file.filename}"
            filepath = UPLOAD_FOLDER / filename
            file.save(filepath)
            saved_paths.append(str(filepath))
            filenames.append(filename)

    from database.mongo import mongodb

    results = moderation_engine.submit_batch(saved_paths)
    moderation_results = []

    for i, result in enumerate(results):
        if not result.get("success", False):
            continue

        filename = filenames[i]
        original_name = file.filename

        prob_fake = result.get("prob_fake", 0)
        prob_real = result.get("prob_real", 0)

        frontend_result = {
            "filename": original_name,
            "is_fake": result.get("is_fake", False),
            "confidence": result.get("confidence", 0.0),
            "prob_real": prob_real,
            "prob_fake": prob_fake,
            "status": "rejected" if result.get("is_fake", False) else "approved",
            "timestamp": datetime.now().isoformat(),
            "risk_assessment": {
                "severity": "CRITICAL" if prob_fake >= 85 else
                "HIGH" if prob_fake >= 70 else
                "MEDIUM" if prob_fake >= 50 else
                "MINIMAL" if prob_real >= 85 else
                "LOW" if prob_real >= 70 else "MEDIUM",
                "description": f"Real: {prob_real}%, Fake: {prob_fake}%",
                "recommendation": get_recommendation(prob_fake, prob_real)
            }
        }

        moderation_results.append({
            "filename": original_name,
            "result": frontend_result
        })

        moderation_data = {
            "filename": filename,
            "is_fake": result.get("is_fake", False),
            "confidence": result.get("confidence", 0.0),
            "prob_real": prob_real,
            "prob_fake": prob_fake,
            "status": "rejected" if result.get("is_fake", False) else "approved",
            "timestamp": datetime.now().isoformat()
        }

        mongodb.save_moderation_result(moderation_data)

    return jsonify({
        "success": True,
        "results": moderation_results
    })



@app.route("/upload", methods=["POST"])
def upload_single():
    if not MODERATION_LOADED:
        return jsonify({"error": "Moderation unavailable"}), 503

    if "file" not in request.files:
        return jsonify({"error": "No file sent"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "No filename"}), 400

    filename = f"{uuid.uuid4()}_{file.filename}"
    filepath = UPLOAD_FOLDER / filename
    file.save(filepath)

    result = moderation_engine.submit_image(str(filepath))
    if not result.get("success", False):
        return jsonify({"error": result.get("error", "Analysis failed")}), 500

    prob_fake = result.get("prob_fake", 0)
    prob_real = result.get("prob_real", 0)

    frontend_result = {
        "filename": file.filename,
        "is_fake": result.get("is_fake", False),
        "confidence": result.get("confidence", 0.0),
        "prob_real": prob_real,
        "prob_fake": prob_fake,
        "status": "rejected" if result.get("is_fake", False) else "approved",
        "timestamp": datetime.now().isoformat(),
        "risk_assessment": {
            "severity": "CRITICAL" if prob_fake >= 85 else
            "HIGH" if prob_fake >= 70 else
            "MEDIUM" if prob_fake >= 50 else
            "MINIMAL" if prob_real >= 85 else
            "LOW" if prob_real >= 70 else "MEDIUM",
            "recommendation": get_recommendation(prob_fake, prob_real)
        }
    }

    from database.mongo import mongodb
    moderation_data = {
        "filename": filename,
        "is_fake": result.get("is_fake", False),
        "confidence": result.get("confidence", 0.0),
        "prob_real": prob_real,
        "prob_fake": prob_fake,
        "status": "rejected" if result.get("is_fake", False) else "approved",
        "timestamp": datetime.now().isoformat()
    }

    mongodb.save_moderation_result(moderation_data)

    return jsonify({
        "success": True,
        "filename": file.filename,
        "result": frontend_result
    })



@app.route("/generate", methods=["POST"])
def generate():
    if not GENERATION_LOADED:
        return jsonify({"error": "Generation unavailable"}), 503

    data = request.get_json() or {}
    prompt = data.get("prompt", "").strip()

    if not prompt:
        return jsonify({"error": "Prompt required"}), 400

    try:
        task_id = generation_engine.submit(prompt=prompt)
        logger.info(f"Generation submitted: {task_id}")
        return jsonify({"success": True, "task_id": task_id})
    except Exception as e:
        logger.error(f"Generation error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/generate/batch", methods=["POST"])
def generate_batch():
    if not GENERATION_LOADED:
        return jsonify({"error": "Generation unavailable"}), 503

    data = request.get_json() or {}
    prompt = data.get("prompt", "").strip()
    count = data.get("count", 1)

    if not prompt:
        return jsonify({"error": "Prompt required"}), 400

    try:
        task_ids = generation_engine.submit_batch(prompt=prompt, count=count)
        logger.info(f"Batch generation submitted: {task_ids}")
        return jsonify({"success": True, "task_ids": task_ids})
    except Exception as e:
        logger.error(f"Batch generation error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/generate/status/<task_id>")
def generation_status(task_id):
    status = generation_engine.get_status(task_id)
    if not status:
        return jsonify({"error": "Task not found"}), 404
    return jsonify(status)


@app.route("/generated/<filename>")
def serve_generated(filename):
    return send_from_directory(GENERATED_FOLDER, filename)


@app.route("/uploads/<filename>")
def serve_uploaded(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route("/heatmaps/<filename>")
def serve_heatmap(filename):
    return send_from_directory(HEATMAP_FOLDER, filename)


@app.route("/stats")
def stats():
    from database.mongo import mongodb
    return jsonify(mongodb.get_stats())


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5001))
    logger.info(f" Server running on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)