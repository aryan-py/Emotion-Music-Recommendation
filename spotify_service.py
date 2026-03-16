"""
Spotify service: loads credentials from .env, fetches emotion playlists at
startup, and caches them as DataFrames that include a spotify_url column.

Falls back gracefully when credentials are absent or API calls fail.
"""

import os
import threading
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# Playlist IDs per emotion index (strip any ?si=... share suffixes)
# TODO: replace the disgusted playlist ID (1) with a proper one —
#       it currently duplicates the sad playlist.
PLAYLIST_IDS = {
    0: "0l9dAmBrUJLylii66JOsHB",  # angry
    1: "6KmSJfEhFvpBM5qzGb5TiL",  # disgusted  ← replace with actual playlist ID
    2: "4cllEPvFdoX6NIVWPKai9I",  # fearful
    3: "0deORnapZgrxFY4nsKr9JA",  # happy
    4: "4kvSlabrnfRCQWfN0MgtgA",  # neutral
    5: "1n6cpWo9ant4WguEo91KZh",  # sad
    6: "37i9dQZEVXbMDoHDwVN2tF",  # surprised
}

_cache: dict[int, pd.DataFrame] = {}
_lock = threading.Lock()
_ready = False


def _build_sp():
    client_id = os.getenv("SPOTIFY_CLIENT_ID", "").strip()
    client_secret = os.getenv("SPOTIFY_CLIENT_SECRET", "").strip()
    if not client_id or not client_secret:
        return None
    return spotipy.Spotify(
        auth_manager=SpotifyClientCredentials(
            client_id=client_id,
            client_secret=client_secret,
        )
    )


def _fetch_playlist(sp: spotipy.Spotify, playlist_id: str) -> pd.DataFrame:
    tracks = []
    results = sp.playlist_tracks(playlist_id, limit=50)
    for item in results.get("items", []):
        t = item.get("track")
        if not t:
            continue
        tracks.append(
            {
                "Name": t["name"],
                "Album": t["album"]["name"],
                "Artist": t["artists"][0]["name"] if t["artists"] else "—",
                "spotify_url": t["external_urls"].get("spotify", ""),
                "track_uri": t.get("uri", ""),
            }
        )
    return pd.DataFrame(tracks)


def _load():
    global _ready
    sp = _build_sp()
    if sp is None:
        print("[spotify_service] No credentials found in .env — using CSV fallback.")
        return

    print("[spotify_service] Fetching Spotify playlists…")
    for idx, pid in PLAYLIST_IDS.items():
        try:
            df = _fetch_playlist(sp, pid)
            with _lock:
                _cache[idx] = df
            print(f"[spotify_service]  ✓ emotion {idx} ({len(df)} tracks)")
        except Exception as exc:
            print(f"[spotify_service]  ✗ emotion {idx} ({pid}): {exc}")

    with _lock:
        _ready = len(_cache) > 0
    if _ready:
        print("[spotify_service] Ready — Spotify data loaded.")


def get_songs(emotion_idx: int) -> pd.DataFrame | None:
    """Return up to 15 rows for the given emotion, or None if not available."""
    with _lock:
        df = _cache.get(emotion_idx)
    if df is not None and not df.empty:
        return df.head(15)
    return None


def is_ready() -> bool:
    with _lock:
        return _ready


_EMOTION_TO_IDX = {
    "angry": 0, "disgusted": 1, "fearful": 2, "happy": 3,
    "neutral": 4, "sad": 5, "surprised": 6,
}


def get_songs_by_name(emotion: str) -> pd.DataFrame | None:
    """Look up songs by emotion name string (case-insensitive)."""
    idx = _EMOTION_TO_IDX.get(emotion.lower())
    if idx is not None:
        return get_songs(idx)
    return None


# Kick off a background fetch immediately on import so the app isn't blocked.
threading.Thread(target=_load, daemon=True).start()
