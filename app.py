"""
app.py  –  Flask REST API for AI Resume Screener
Endpoints:
  GET  /           → Web UI
  POST /predict    → JSON or multipart (PDF/TXT file) → prediction
  GET  /health     → health check
"""

import os
import re
import pickle
import PyPDF2
import io
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024   # 5 MB limit

ALLOWED_EXTENSIONS = {"pdf", "txt", "doc"}
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "resume_classifier.pkl")

# ── Load model ─────────────────────────────────────────────────────────────

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run `python train_model.py` first."
        )
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

try:
    MODEL = load_model()
    print("✓ Model loaded successfully.")
except FileNotFoundError as e:
    print(f"⚠ {e}")
    MODEL = None

# ── Helpers ────────────────────────────────────────────────────────────────

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s\+\#]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_text_from_pdf(file_bytes):
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        return None

def predict_resume(text):
    if MODEL is None:
        return None, None, None
    cleaned = clean_text(text)
    prediction = MODEL.predict([cleaned])[0]
    probabilities = MODEL.predict_proba([cleaned])[0]
    classes = MODEL.classes_
    prob_dict = {cls: round(float(prob) * 100, 2) for cls, prob in zip(classes, probabilities)}
    confidence = round(float(max(probabilities)) * 100, 2)
    return prediction, confidence, prob_dict

# ── Routes ─────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": MODEL is not None,
        "categories": list(MODEL.classes_) if MODEL else []
    })

@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts:
      - JSON body: {"resume_text": "..."}
      - Multipart form: file (PDF or TXT)
    Returns:
      {
        "category": "Data Scientist",
        "confidence": 92.4,
        "all_probabilities": { "Data Scientist": 92.4, ... },
        "status": "success"
      }
    """
    if MODEL is None:
        return jsonify({
            "error": "Model not loaded. Run `python train_model.py` first.",
            "status": "error"
        }), 503

    resume_text = None

    # Case 1: JSON body
    if request.is_json:
        data = request.get_json()
        resume_text = data.get("resume_text", "").strip()
        if not resume_text:
            return jsonify({"error": "Field 'resume_text' is empty.", "status": "error"}), 400

    # Case 2: File upload
    elif "file" in request.files:
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected.", "status": "error"}), 400
        if not allowed_file(file.filename):
            return jsonify({"error": "File type not supported. Upload PDF or TXT.", "status": "error"}), 400

        file_bytes = file.read()
        ext = file.filename.rsplit(".", 1)[1].lower()

        if ext == "pdf":
            resume_text = extract_text_from_pdf(file_bytes)
            if not resume_text:
                return jsonify({"error": "Could not extract text from PDF.", "status": "error"}), 400
        else:
            try:
                resume_text = file_bytes.decode("utf-8", errors="ignore")
            except Exception:
                return jsonify({"error": "Could not read file.", "status": "error"}), 400

    # Case 3: Form text field
    elif "resume_text" in request.form:
        resume_text = request.form.get("resume_text", "").strip()

    else:
        return jsonify({
            "error": "Send JSON with 'resume_text' or upload a file.",
            "status": "error"
        }), 400

    if len(resume_text.split()) < 10:
        return jsonify({"error": "Resume text too short. Please provide more content.", "status": "error"}), 400

    category, confidence, all_probs = predict_resume(resume_text)

    return jsonify({
        "category": category,
        "confidence": confidence,
        "all_probabilities": all_probs,
        "word_count": len(resume_text.split()),
        "status": "success"
    })

# ── Run ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_ENV", "production") == "development"
    print(f"Starting AI Resume Screener on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=debug)
