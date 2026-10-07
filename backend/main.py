import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from emotion_service import EmotionDetector
from spotify_service import recommend

app = FastAPI(title="Emotion Music Recommender")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

detector = EmotionDetector()


@app.get("/api/health")
def health():
    return {"status": "ok", "classes": detector.class_names}


@app.post("/api/detect")
def detect(file: UploadFile = File(...)):
    data = np.frombuffer(file.file.read(), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(400, "Invalid image")
    result = detector.detect(image)
    if result is None:
        return {"face": False}
    return {"face": True, **result}


@app.get("/api/recommend")
def get_recommendations(emotion: str, mode: str = "match"):
    try:
        return {"emotion": emotion, "mode": mode, "tracks": recommend(emotion, mode)}
    except FileNotFoundError:
        raise HTTPException(500, "muse_v3.csv not found. Put it in backend\\data\\")