# Real-Time Emotion-Based Music Recommender

A web app that reads your facial emotion through the webcam and recommends songs that match your mood, in real time.

The React front end captures webcam frames and sends them to a **FastAPI** backend. The backend detects the face with OpenCV, crops it, and passes it to an **EfficientNetB0** model trained on FER2013 to predict one of 7 emotions. The emotion is smoothed over several frames so a single odd frame doesn't change the music. Songs are then picked from a 60,000+ track mood dataset and played through **Spotify** embeds.

## Demo



https://github.com/user-attachments/assets/72535ea1-b2f2-48c0-aa08-c3388ca49ef2




## Features

- Live webcam emotion tracking with a face bounding box and per-emotion probability bars
- 7 emotions: angry, disgust, fear, happy, neutral, sad, surprise
- Frame smoothing (majority vote over the last 7 frames) so the music doesn't flicker
- Two music modes: **Match my mood** and **Lift my mood**
- Songs ranked by valence and arousal scores from the MuSe dataset
- Spotify embedded player with a clickable track list and a Refresh button
- Dark, responsive React interface

## How it works

1. The browser captures a webcam frame about once per second and sends it to `POST /api/detect`.
2. OpenCV (Haar cascade) finds the largest face and crops it.
3. The cropped face is resized to 96x96 and passed to EfficientNetB0, which outputs probabilities for the 7 emotions.
4. The front end keeps the last 7 predictions. When one emotion appears 4 or more times, it becomes the stable emotion.
5. `GET /api/recommend` maps the emotion to a target point on the valence (how positive) and arousal (how energetic) scale, then returns the closest songs from `muse_v3.csv`, shuffled for variety.
6. Songs play through Spotify's embed player.

## Tech stack

| Layer | Technology |
|---|---|
| Front end | React 18, Vite |
| Backend API | FastAPI, Uvicorn |
| Emotion model | EfficientNetB0 (transfer learning), TensorFlow / Keras |
| Face detection | OpenCV Haar cascade |
| Music data | MuSe dataset (`muse_v3.csv`), pandas, NumPy |
| Playback | Spotify embed player |
| Training data | FER2013 |

## Project structure

```
emotion-music/
├── README.md
├── .gitignore
├── backend/
│   ├── main.py               FastAPI app and API endpoints
│   ├── emotion_service.py    face detection and emotion prediction
│   ├── spotify_service.py    song recommendation from muse_v3.csv
│   ├── train.py              EfficientNetB0 training script
│   ├── requirements.txt      Python dependencies
│   ├── data/                 muse_v3.csv, train/ and test/ (FER2013) - not in git
│   └── models/               emotion_model.keras, class_names.json - created by train.py
└── frontend/
    ├── index.html
    ├── package.json
    ├── vite.config.js        dev server and /api proxy to the backend
    └── src/
        ├── main.jsx
        ├── App.jsx           webcam capture, live detection, music UI
        ├── App.css
        └── index.css
```

`node_modules/` and `package-lock.json` appear in `frontend/` after `npm install`.

## Setup (Windows)

Requirements: **Python 3.11** (TensorFlow 2.16 does not support Python 3.14), **Node.js 20+**, and a webcam.

### 1. Get the data

- **FER2013**: download from Kaggle ("msambare/fer2013") and extract so these exist:
  `backend/data/train/<7 emotion folders>` and `backend/data/test/<7 emotion folders>`
- **MuSe dataset**: place `muse_v3.csv` in `backend/data/`

### 2. Backend environment and training

The virtual environment is created at a short path outside the project (`C:\ev`). This keeps the project folder small and avoids Windows long-path and OneDrive sync errors when TensorFlow installs.

```bat
py -3.11 -m venv C:\ev
C:\ev\Scripts\activate.bat
python -m pip install --upgrade pip
pip install --no-cache-dir -r backend\requirements.txt
cd backend
python train.py
```

Training saves `backend/models/emotion_model.keras`. It runs on CPU, so it can take an hour or more. You only need to train once.

### 3. Run the backend

```bat
cd backend
C:\ev\Scripts\activate.bat
uvicorn main:app --port 8000
```

Check `http://127.0.0.1:8000/api/health`.

### 4. Run the front end

Open a second terminal:

```bat
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`, click **Start camera**, allow permission, and hold an expression for a few seconds. `npm install` is only needed the first time.

## API

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/health` | Status and emotion class names |
| POST | `/api/detect` | Upload a frame, returns emotion, confidence, probabilities and face box |
| GET | `/api/recommend?emotion=happy&mode=match` | Returns 9 songs. `mode` is `match` or `uplift` |

## Model notes

- Input: 96x96 face crop, EfficientNetB0 with ImageNet weights
- Training: phase 1 trains the new classification head, phase 2 fine-tunes the top 40 layers
- Class weights soften the imbalance of the small "disgust" class
- Validation accuracy: **XX%** on the FER2013 test split (replace with your own result)

## Limitations

- FER2013 labels are noisy, so accuracy of about 60-68% is typical for this dataset. Good lighting and a clear, front-facing expression give the best results.
- Haar cascade detection works best on frontal faces.
- Recommendations come from tag-based valence/arousal scores, so mood matches are approximate.
- The dataset is from around 2019, so a few tracks may no longer be on Spotify.
- Spotify embeds play full tracks only when you are logged in to Spotify in the same browser. Otherwise you hear a short preview.
- The face model reflects the biases of FER2013 and may be less accurate for some people. It is a demo, not a clinical or psychological tool.
- Webcam frames are sent only to your own local backend and are not stored.

## Future improvements

- Face landmark detection (MediaPipe) in place of the Haar cascade
- Live Spotify Web API recommendations and playlist creation
- Per-user history and mood trends
- Cloud deployment with Docker

## Author

- **Chopperla Sathvika** - [GitHub](https://github.com/Sathvika046) | [LinkedIn](https://www.linkedin.com/in/chopperla-sathvika-7702b0331)

## Acknowledgements

- FER2013 facial expression dataset
- MuSe (Music Sentiment) dataset
- TensorFlow, FastAPI, React, Vite, OpenCV
