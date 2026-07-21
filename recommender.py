"""Hybrid movie recommendation engine.

This module provides a reusable recommendation engine that combines:
- content similarity from genre and overview text features
- popularity/rating weighting to avoid purely plot-based matches
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import math


class HybridRecommender:
    """Recommend movies using a hybrid score that mixes similarity and popularity.

    Args:
        movies: List of movie dictionaries with at least title, overview, genres,
            vote_average, popularity, and sentiment attributes.
    """

    def __init__(self, movies: List[Dict]) -> None:
        self.movies = movies
        self._movie_index = {movie["title"].lower(): movie for movie in movies}

    def _normalize(self, value: float, min_value: float, max_value: float) -> float:
        if max_value <= min_value:
            return 0.0
        return (value - min_value) / (max_value - min_value)

    def _text_similarity(self, text_a: str, text_b: str) -> float:
        if not text_a or not text_b:
            return 0.0
        words_a = set(text_a.lower().split())
        words_b = set(text_b.lower().split())
        if not words_a or not words_b:
            return 0.0
        overlap = len(words_a & words_b)
        return overlap / max(len(words_a | words_b), 1)

    def _genre_similarity(self, genres_a: List[str], genres_b: List[str]) -> float:
        if not genres_a or not genres_b:
            return 0.0
        shared = set(genres_a) & set(genres_b)
        return len(shared) / max(len(set(genres_a) | set(genres_b)), 1)

    def _combined_similarity(self, movie_a: Dict, movie_b: Dict) -> float:
        text_score = self._text_similarity(movie_a.get("overview", ""), movie_b.get("overview", ""))
        genre_score = self._genre_similarity(movie_a.get("genres", []), movie_b.get("genres", []))
        sentiment_score = 1.0 - abs(float(movie_a.get("sentiment", 0.0)) - float(movie_b.get("sentiment", 0.0)))
        return 0.45 * text_score + 0.35 * genre_score + 0.20 * sentiment_score

    def recommend(self, title: str, top_n: int = 5) -> List[Dict]:
        """Return ranked recommendations for the given movie title.

        Args:
            title: The title of the reference movie.
            top_n: Number of recommendations to return.

        Returns:
            A list of movie dictionaries with an added ``score`` field.
        """
        reference = self._movie_index.get(title.lower())
        if not reference:
            return []

        ratings = []
        for movie in self.movies:
            if movie["title"].lower() == title.lower():
                continue
            similarity = self._combined_similarity(reference, movie)
            rating_score = self._normalize(float(movie.get("vote_average", 0.0)), 5.0, 10.0)
            popularity_score = self._normalize(float(movie.get("popularity", 0.0)), 0.0, 100.0)
            score = (0.7 * similarity) + (0.2 * rating_score) + (0.1 * popularity_score)
            ratings.append({**movie, "score": round(score, 4)})

        ratings.sort(key=lambda item: item["score"], reverse=True)
        return ratings[:top_n]
