import os
import json
import threading
import base64
from datetime import datetime
import pandas as pd
import spotipy
from flask import Flask, render_template, jsonify, request, redirect, session
from dotenv import load_dotenv
from spotipy.oauth2 import SpotifyOAuth
from camera import process_frame, music_rec, music_dist, emotion_dict
import spotify_service
import openai_service

load_dotenv()

app = Flask(__name__)

_secret_key = os.getenv("FLASK_SECRET_KEY", "")
if not _secret_key:
    import warnings
    warnings.warn(
        "FLASK_SECRET_KEY is not set in .env — Spotify OAuth sessions will not "
        "persist across restarts. Add FLASK_SECRET_KEY=<random-string> to .env.",
        stacklevel=1,
    )
    _secret_key = "dev-secret-change-me"
app.secret_key = _secret_key

SPOTIFY_SCOPE = "playlist-modify-public playlist-modify-private"

# Derived from camera.py's music_dist — single source of truth, no duplication.
_EMOTION_CSV = {emotion_dict[k]: v for k, v in music_dist.items()}


def _sp_oauth():
    return SpotifyOAuth(
        client_id=os.getenv("SPOTIFY_CLIENT_ID", "").strip(),
        client_secret=os.getenv("SPOTIFY_CLIENT_SECRET", "").strip(),
        redirect_uri=os.getenv("SPOTIFY_REDIRECT_URI", "http://127.0.0.1:5000/auth/callback"),
        scope=SPOTIFY_SCOPE,
        cache_handler=spotipy.cache_handler.FlaskSessionCacheHandler(session),
        show_dialog=False,
    )


def _songs_for_emotion(emotion: str) -> list:
    songs_df = spotify_service.get_songs_by_name(emotion)
    if songs_df is None:
        csv_path = _EMOTION_CSV.get(emotion, music_dist[4])
        df = pd.read_csv(csv_path)
        songs_df = df[["Name", "Album", "Artist"]].copy()
        songs_df["spotify_url"] = ""
        songs_df["track_uri"] = ""
    return songs_df.head(15).to_dict(orient="records")


# ── Pages ────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


# ── Face scan ────────────────────────────────────────────────

@app.route("/process_frame", methods=["POST"])
def process_frame_route():
    data = request.get_json(silent=True)
    if not data or "frame" not in data:
        return jsonify({"error": "Missing frame data"}), 400
    frame_data = data["frame"]
    if "," not in frame_data:
        return jsonify({"error": "Invalid frame format — expected a data URL"}), 400
    _, encoded = frame_data.split(",", 1)
    try:
        image_bytes = base64.b64decode(encoded)
    except Exception:
        return jsonify({"error": "Invalid base64 encoding"}), 400
    faces, emotion, songs_df = process_frame(image_bytes)
    return jsonify({
        "faces": faces,
        "emotion": emotion,
        "songs": songs_df.to_dict(orient="records"),
    })


# ── Voice chat ───────────────────────────────────────────────

@app.route("/voice_chat", methods=["POST"])
def voice_chat():
    """Accepts either:
      {"text": "..."}           – transcript already obtained (e.g. Web Speech API)
      {"audio": "<data-url>"}   – raw audio to transcribe via Sarvam STT
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Missing request body"}), 400

    # ── Resolve transcript ────────────────────────────────────
    if "text" in data:
        transcript = data["text"].strip()
        print(f"[voice_chat] text received: {repr(transcript)}")
    elif "audio" in data:
        audio_data = data["audio"]
        if "," not in audio_data:
            return jsonify({"error": "Invalid audio format"}), 400
        header, b64 = audio_data.split(",", 1)
        mime = header.split(":")[1].split(";")[0]
        ext  = mime.split("/")[1].split(";")[0]
        try:
            audio_bytes = base64.b64decode(b64)
        except Exception:
            return jsonify({"error": "Invalid audio encoding"}), 400
        print(f"[voice_chat] audio received: {len(audio_bytes)} bytes, ext={ext}")
        try:
            transcript = openai_service.speech_to_text(audio_bytes, f"audio.{ext}")
            print(f"[voice_chat] transcript: {repr(transcript)}")
        except Exception as exc:
            print(f"[voice_chat] STT failed: {exc}")
            return jsonify({"error": f"Speech recognition failed: {exc}"}), 500
    else:
        return jsonify({"error": "Provide either 'text' or 'audio'"}), 400

    if not transcript.strip():
        return jsonify({"error": "Could not understand — please try again."}), 422

    # ── Mood analysis ─────────────────────────────────────────
    mood = openai_service.analyze_mood(transcript)
    emotion       = mood.get("emotion", "Neutral")
    tips          = mood.get("tips", [])
    response_text = mood.get("response", "")
    print(f"[voice_chat] emotion={emotion}")

    try:
        songs = _songs_for_emotion(emotion)
        print(f"[voice_chat] songs returned: {len(songs)}")
    except Exception as exc:
        print(f"[voice_chat] _songs_for_emotion failed: {exc}")
        songs = []

    return jsonify({
        "transcript":    transcript,
        "emotion":       emotion,
        "tips":          tips,
        "response_text": response_text,
        "songs":         songs,
    })


# ── Journal ──────────────────────────────────────────────────

_JOURNAL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "journal_history.json")
_journal_lock = threading.Lock()


def _load_journal_history():
    if not os.path.exists(_JOURNAL_FILE):
        return []
    try:
        with open(_JOURNAL_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return []


def _append_journal_entry(entry: dict):
    with _journal_lock:
        history = _load_journal_history()
        history.insert(0, entry)
        with open(_JOURNAL_FILE, "w") as f:
            json.dump(history[:100], f, indent=2)


@app.route("/journal", methods=["POST"])
def journal():
    data = request.get_json(silent=True)
    if not data or "text" not in data:
        return jsonify({"error": "Missing journal text"}), 400

    text = data["text"].strip()
    if len(text) < 20:
        return jsonify({"error": "Write a bit more before analysing."}), 422

    mood = openai_service.analyze_mood(text)
    emotion      = mood.get("emotion", "Neutral")
    tips         = mood.get("tips", [])
    response_text = mood.get("response", "")

    try:
        songs = _songs_for_emotion(emotion)
    except Exception as exc:
        print(f"[journal] _songs_for_emotion failed: {exc}")
        songs = []

    entry = {
        "id":      str(int(datetime.now().timestamp() * 1000)),
        "date":    datetime.now().strftime("%b %d, %Y · %I:%M %p"),
        "preview": text[:120],
        "emotion": emotion,
        "tips":    tips,
        "songs":   songs[:6],
    }
    _append_journal_entry(entry)

    return jsonify({
        "emotion":  emotion,
        "tips":     tips,
        "response": response_text,
        "songs":    songs,
    })


@app.route("/journal/history", methods=["GET"])
def journal_history():
    return jsonify(_load_journal_history())


# ── Spotify playlist creation ────────────────────────────────

@app.route("/create_playlist", methods=["POST"])
def create_playlist():
    data = request.get_json()
    emotion = data.get("emotion", "Mixed")
    track_uris = [u for u in data.get("track_uris", []) if u]

    oauth = _sp_oauth()
    token_info = oauth.get_cached_token()

    if not token_info:
        session["pending_emotion"] = emotion
        session["pending_uris"] = track_uris
        return jsonify({"auth_url": oauth.get_authorize_url()})

    sp = spotipy.Spotify(auth=token_info["access_token"])
    playlist_url = _make_playlist(sp, emotion, track_uris)
    return jsonify({"playlist_url": playlist_url})


@app.route("/auth/callback")
def auth_callback():
    code = request.args.get("code")
    if not code:
        return redirect("/")

    oauth = _sp_oauth()
    token_info = oauth.get_access_token(code, check_cache=False)

    emotion = session.pop("pending_emotion", None)
    uris = session.pop("pending_uris", [])

    if emotion and uris:
        sp = spotipy.Spotify(auth=token_info["access_token"])
        playlist_url = _make_playlist(sp, emotion, uris)
        return redirect(f"/?playlist_url={playlist_url}")

    return redirect("/")


def _make_playlist(sp: spotipy.Spotify, emotion: str, track_uris: list) -> str:
    user_id = sp.me()["id"]
    playlist = sp.user_playlist_create(
        user_id,
        f"MoodTunes – {emotion}",
        public=False,
        description=f"Auto-generated by MoodTunes for {emotion} mood",
    )
    if track_uris:
        sp.playlist_add_items(playlist["id"], track_uris)
    return playlist["external_urls"]["spotify"]


@app.route("/t")
def gen_table():
    return music_rec().to_json(orient="records")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
