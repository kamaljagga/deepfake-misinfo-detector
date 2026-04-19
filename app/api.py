from flask import Flask, request, jsonify
from flask_cors import CORS
import tempfile, os, sys
sys.path.append("..")

from module_a_deepfake.frame_extractor import extract_frames
from module_a_deepfake.face_detector import FaceDetector
from module_a_deepfake.classifier import DeepfakeClassifier
from module_a_deepfake.aggregator import aggregate_frame_predictions
from module_b_misinfo.model import MisinfoDetector

app = Flask(__name__)
CORS(app)  # Allow requests from any frontend

# Load models once at startup
face_det     = FaceDetector()
deepfake_clf = DeepfakeClassifier("../models/deepfake/efficientnet_b0.pth")
misinfo_clf  = MisinfoDetector("../models/nlp/distilbert_finetuned.pth")

@app.route("/api/analyze/video", methods=["POST"])
def analyze_video():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, file.filename)
        file.save(path)

        frames = extract_frames(path, os.path.join(tmpdir, "frames"), fps=2)
        preds  = [deepfake_clf.predict(face_det.extract_face(f)) for f in frames]
        result = aggregate_frame_predictions(preds)

    return jsonify(result)

@app.route("/api/analyze/text", methods=["POST"])
def analyze_text():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400

    result = misinfo_clf.predict(data["text"])
    return jsonify(result)

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "models_loaded": True})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)