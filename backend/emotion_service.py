import json
from pathlib import Path

import cv2
import numpy as np
from tensorflow import keras

MODEL_DIR = Path(__file__).parent / "models"
IMG = 96  # must match train.py


class EmotionDetector:
    def __init__(self):
        self.model = keras.models.load_model(MODEL_DIR / "emotion_model.keras")
        self.class_names = json.loads((MODEL_DIR / "class_names.json").read_text())
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    def detect(self, bgr_image):
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60)
        )
        if len(faces) == 0:
            return None

        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])  # largest face
        crop = cv2.resize(gray[y:y + h, x:x + w], (IMG, IMG))
        rgb = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB).astype("float32")

        probs = self.model(rgb[None, ...], training=False).numpy()[0]
        idx = int(np.argmax(probs))
        return {
            "emotion": self.class_names[idx],
            "confidence": float(probs[idx]),
            "probabilities": {n: float(p) for n, p in zip(self.class_names, probs)},
            "box": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
        }