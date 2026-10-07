from pathlib import Path

import numpy as np
import pandas as pd

CSV_PATH = Path(__file__).parent / "data" / "muse_v3.csv"

# (valence, arousal) targets on the dataset's ~1-9 scale
#   valence = how positive the song feels, arousal = how energetic it is
TARGETS = {
    "match": {
        "happy":    (7.0, 5.5),
        "sad":      (3.0, 3.0),
        "angry":    (3.0, 6.3),
        "fear":     (3.5, 5.5),
        "surprise": (6.5, 6.0),
        "disgust":  (3.5, 5.0),
        "neutral":  (5.5, 4.3),
    },
    "uplift": {
        "happy":    (7.0, 5.5),
        "sad":      (6.5, 4.8),
        "angry":    (6.0, 3.0),
        "fear":     (6.0, 3.2),
        "surprise": (6.5, 5.5),
        "disgust":  (6.0, 4.0),
        "neutral":  (6.8, 5.3),
    },
}

_songs = None


def _load():
    global _songs
    if _songs is None:
        df = pd.read_csv(
            CSV_PATH,
            usecols=["track", "artist", "valence_tags", "arousal_tags", "spotify_id", "genre"],
        )
        df = df.dropna(subset=["spotify_id", "track", "artist"])
        df = df.drop_duplicates(subset="spotify_id").reset_index(drop=True)
        _songs = df
    return _songs


def recommend(emotion, mode="match", count=9, pool=150):
    df = _load()
    group = TARGETS.get(mode, TARGETS["match"])
    v, a = group.get(emotion, group["neutral"])

    dist = np.sqrt((df["valence_tags"] - v) ** 2 + (df["arousal_tags"] - a) ** 2)
    nearest = df.loc[dist.nsmallest(pool).index]
    picks = nearest.sample(n=min(count, len(nearest)))

    return [
        {
            "id": row.spotify_id,
            "name": row.track,
            "artists": row.artist,
            "album": str(row.genre),
            "image": None,
            "url": f"https://open.spotify.com/track/{row.spotify_id}",
        }
        for row in picks.itertuples()
    ]