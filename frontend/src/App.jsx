import { useCallback, useEffect, useRef, useState } from "react";
import "./App.css";

const EMOJI = {
  happy: "😄", sad: "😢", angry: "😠", fear: "😨",
  surprise: "😮", disgust: "🤢", neutral: "😐",
};
const WINDOW = 7; // frames used for smoothing
const NEEDED = 4; // same emotion this many times in the window = stable

export default function App() {
  const videoRef = useRef(null);
  const overlayRef = useRef(null);
  const captureRef = useRef(document.createElement("canvas"));
  const streamRef = useRef(null);
  const historyRef = useRef([]);
  const busyRef = useRef(false);

  const [running, setRunning] = useState(false);
  const [current, setCurrent] = useState(null);
  const [stable, setStable] = useState(null);
  const [mode, setMode] = useState("match");
  const [tracks, setTracks] = useState([]);
  const [playingId, setPlayingId] = useState(null);
  const [loadingTracks, setLoadingTracks] = useState(false);
  const [refreshKey, setRefreshKey] = useState(0);
  const [error, setError] = useState("");

  const start = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 },
      });
      streamRef.current = stream;
      videoRef.current.srcObject = stream;
      await videoRef.current.play();
      historyRef.current = [];
      setStable(null);
      setError("");
      setRunning(true);
    } catch {
      setError("Could not access the webcam. Allow camera permission and retry.");
    }
  };

  const stop = () => {
    setRunning(false);
    streamRef.current?.getTracks().forEach((t) => t.stop());
    setCurrent(null);
    const o = overlayRef.current;
    if (o) o.getContext("2d").clearRect(0, 0, o.width, o.height);
  };

  const drawBox = (box, w, h) => {
    const o = overlayRef.current;
    o.width = w;
    o.height = h;
    const ctx = o.getContext("2d");
    ctx.clearRect(0, 0, w, h);
    if (!box) return;
    ctx.strokeStyle = "#1db954";
    ctx.lineWidth = 3;
    ctx.strokeRect(box.x, box.y, box.w, box.h);
  };

  const tick = useCallback(async () => {
    const v = videoRef.current;
    if (!v || v.videoWidth === 0 || busyRef.current) return;
    busyRef.current = true;
    try {
      const c = captureRef.current;
      c.width = v.videoWidth;
      c.height = v.videoHeight;
      c.getContext("2d").drawImage(v, 0, 0);
      const blob = await new Promise((r) => c.toBlob(r, "image/jpeg", 0.8));
      const fd = new FormData();
      fd.append("file", blob, "frame.jpg");

      const res = await fetch("/api/detect", { method: "POST", body: fd });
      if (!res.ok) throw new Error("detect failed");
      const data = await res.json();
      setError("");

      if (data.face) {
        drawBox(data.box, c.width, c.height);
        setCurrent(data);
        const h = historyRef.current;
        h.push(data.emotion);
        if (h.length > WINDOW) h.shift();
        const counts = {};
        h.forEach((e) => (counts[e] = (counts[e] || 0) + 1));
        const [top, n] = Object.entries(counts).sort((a, b) => b[1] - a[1])[0];
        if (n >= NEEDED) setStable(top);
      } else {
        drawBox(null, c.width, c.height);
        setCurrent(null);
      }
    } catch {
      setError("Cannot reach the backend. Is uvicorn running on port 8000?");
    } finally {
      busyRef.current = false;
    }
  }, []);

  useEffect(() => {
    if (!running) return;
    const id = setInterval(tick, 900);
    return () => clearInterval(id);
  }, [running, tick]);

  useEffect(() => {
    if (!stable) return;
    let cancelled = false;
    (async () => {
      setLoadingTracks(true);
      try {
        const res = await fetch(`/api/recommend?emotion=${stable}&mode=${mode}`);
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "Recommendation failed");
        if (!cancelled) {
          setTracks(data.tracks);
          setPlayingId(data.tracks[0]?.id ?? null);
          setError("");
        }
      } catch (e) {
        if (!cancelled) setError(e.message);
      } finally {
        if (!cancelled) setLoadingTracks(false);
      }
    })();
    return () => { cancelled = true; };
  }, [stable, mode, refreshKey]);

  const probs = current
    ? Object.entries(current.probabilities).sort((a, b) => b[1] - a[1])
    : [];

  return (
    <div className="app">
      <header>
        <h1>🎧 Emotion Music Recommender</h1>
        <p>Your face sets the mood. Spotify plays the songs.</p>
      </header>

      {error && <div className="error">{error}</div>}

      <main>
        <section className="panel">
          <div className="video-wrap">
            <video ref={videoRef} muted playsInline />
            <canvas ref={overlayRef} />
            {!running && <div className="placeholder">Camera is off</div>}
          </div>

          <div className="controls">
            {running ? (
              <button className="btn stop" onClick={stop}>Stop camera</button>
            ) : (
              <button className="btn" onClick={start}>Start camera</button>
            )}
            <div className="toggle">
              <button className={mode === "match" ? "on" : ""} onClick={() => setMode("match")}>
                Match my mood
              </button>
              <button className={mode === "uplift" ? "on" : ""} onClick={() => setMode("uplift")}>
                Lift my mood
              </button>
            </div>
          </div>

          <div className="emotion">
            {current ? (
              <>
                <div className="big">
                  {EMOJI[current.emotion]} {current.emotion}
                  <span> {(current.confidence * 100).toFixed(0)}%</span>
                </div>
                {probs.map(([name, p]) => (
                  <div className="bar-row" key={name}>
                    <label>{name}</label>
                    <div className="bar"><div style={{ width: `${p * 100}%` }} /></div>
                  </div>
                ))}
              </>
            ) : (
              <p className="muted">
                {running ? "Looking for a face…" : "Start the camera to detect your emotion."}
              </p>
            )}
          </div>
        </section>

        <section className="panel">
          <div className="rec-head">
            <h2>{stable ? `Songs for ${EMOJI[stable]} ${stable}` : "Recommendations"}</h2>
            {stable && (
              <button className="btn small" onClick={() => setRefreshKey((k) => k + 1)}>
                Refresh
              </button>
            )}
          </div>

          {playingId && (
            <iframe
              key={playingId}
              title="Spotify player"
              src={`https://open.spotify.com/embed/track/${playingId}`}
              width="100%"
              height="152"
              allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture"
              loading="lazy"
            />
          )}

          {loadingTracks && <p className="muted">Finding songs…</p>}

          <div className="tracks">
            {tracks.map((t) => (
              <button
                key={t.id}
                className={`track ${t.id === playingId ? "active" : ""}`}
                onClick={() => setPlayingId(t.id)}
              >
                {t.image && <img src={t.image} alt="" />}
                <div>
                  <strong>{t.name}</strong>
                  <span>{t.artists}</span>
                </div>
              </button>
            ))}
          </div>
          {!stable && (
            <p className="muted">
              Songs appear after your emotion is detected consistently for a few seconds.
            </p>
          )}
        </section>
      </main>
    </div>
  );
}