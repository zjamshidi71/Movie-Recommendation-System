"""TMDB data fetching and local caching utilities.

This module loads movie metadata from TMDB when available, stores the results locally,
and provides a safe fallback for offline or rate-limited sessions.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import requests

DATA_DIR = Path(__file__).parent / "data"
CACHE_FILE = DATA_DIR / "cache" / "tmdb_movies.json"


class TMDBClient:
    """Small wrapper around the TMDB API with local caching."""

    def __init__(self, api_key: Optional[str] = None, cache_file: Optional[Path] = None) -> None:
        self.api_key = api_key or os.getenv("TMDB_API_KEY")
        self.cache_file = cache_file or CACHE_FILE
        self.base_url = "https://api.themoviedb.org/3"
        self.session = requests.Session()
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)

    def _load_cache(self) -> List[Dict]:
        if not self.cache_file.exists():
            return []
        try:
            with self.cache_file.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        except (json.JSONDecodeError, OSError):
            return []

    def _save_cache(self, movies: List[Dict]) -> None:
        try:
            with self.cache_file.open("w", encoding="utf-8") as fh:
                json.dump(movies, fh, indent=2)
        except OSError:
            pass

    def fetch_popular_movies(self, limit: int = 80) -> List[Dict]:
        """Fetch popular movies from TMDB and cache the results.

        Args:
            limit: Number of movies to request.

        Returns:
            A list of movie dictionaries enriched with metadata.
        """
        cached = self._load_cache()
        if cached:
            return cached[:limit]

        if not self.api_key:
            return []

        url = f"{self.base_url}/movie/popular"
        params = {"api_key": self.api_key, "language": "en-US", "page": 1}
        try:
            response = self.session.get(url, params=params, timeout=15)
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException:
            return []

        movies: List[Dict] = []
        for item in payload.get("results", [])[:limit]:
            try:
                movie_id = item["id"]
                details = self._fetch_details(movie_id)
            except Exception:
                details = {}

            movie = {
                "id": item.get("id"),
                "title": item.get("title"),
                "overview": item.get("overview") or "",
                "genres": [genre.get("name") for genre in details.get("genres", []) if genre.get("name")],
                "vote_average": item.get("vote_average", 0.0),
                "popularity": item.get("popularity", 0.0),
                "release_date": item.get("release_date") or details.get("release_date") or "",
                "poster_path": item.get("poster_path") or "",
                "director": details.get("director", ""),
                "cast": details.get("cast", []),
                "revenue": details.get("revenue"),
                "budget": details.get("budget"),
                "runtime": details.get("runtime"),
            }
            movies.append(movie)

        self._save_cache(movies)
        return movies

    def _fetch_details(self, movie_id: int) -> Dict:
        """Fetch a single movie's details and credits.

        Args:
            movie_id: TMDB movie identifier.

        Returns:
            A dictionary of enriched metadata.
        """
        details_url = f"{self.base_url}/movie/{movie_id}"
        credits_url = f"{self.base_url}/movie/{movie_id}/credits"
        params = {"api_key": self.api_key, "language": "en-US"}

        details_payload = {}
        credits_payload = {}
        try:
            details_response = self.session.get(details_url, params=params, timeout=15)
            details_response.raise_for_status()
            details_payload = details_response.json()

            credits_response = self.session.get(credits_url, params=params, timeout=15)
            credits_response.raise_for_status()
            credits_payload = credits_response.json()
        except requests.RequestException:
            return {}

        cast = []
        for member in credits_payload.get("cast", [])[:6]:
            cast.append(member.get("name", ""))

        director = ""
        for crew_member in credits_payload.get("crew", []):
            if crew_member.get("job") == "Director":
                director = crew_member.get("name", "")
                break

        return {
            "genres": details_payload.get("genres", []),
            "release_date": details_payload.get("release_date"),
            "director": director,
            "cast": cast,
            "revenue": details_payload.get("revenue"),
            "budget": details_payload.get("budget"),
            "runtime": details_payload.get("runtime"),
        }
