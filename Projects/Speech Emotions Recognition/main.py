from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import os
import pickle
import librosa
import uuid

app = Flask(__name__, static_folder='static')
CORS(app, resources={r"/*": {"origins": "*"}})

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model and label encoder
with open("emotion_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# MFCC extraction (SAME as training)
def extract_mfcc(file_path):
    y, sr = librosa.load(file_path, duration=3, offset=0.5)

    if len(y) < sr * 3:
        y = np.pad(y, (0, sr*3 - len(y)))

    mfcc = np.mean(
        librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T,
        axis=0
    )
    return mfcc


@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "Speech Emotion Recognition API running"})


@app.route("/predict", methods=["POST"])
def predict():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    file = request.files["audio"]
    filename = f"{uuid.uuid4()}.wav"
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    try:
        features = extract_mfcc(file_path)
        features = features.reshape(1, -1)

        prediction = model.predict(features)[0]
        emotion = label_encoder.inverse_transform([prediction])[0]

        return jsonify({"emotion": emotion})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        os.remove(file_path)

# Run Flask
if __name__ == '__main__':
    app.run(host="127.0.0.1", port=5000, debug=True)