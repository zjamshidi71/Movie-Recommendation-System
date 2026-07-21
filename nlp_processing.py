"""Lightweight NLP enrichment for movie descriptions.

The recommender uses these features as input to scoring. In particular:
- sentiment gives a rough tone signal so comedy or upbeat films can be favored
  when the user asks for a lighter mood.
- TF-IDF-style overlap over overview text provides semantic similarity between
  movies without requiring a heavyweight embedding model.
"""

from __future__ import annotations

import re
from typing import Dict, List


def normalize_text(text: str) -> str:
    """Normalize whitespace and punctuation for text matching."""
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def extract_features(movies: List[Dict]) -> List[Dict]:
    """Enrich each movie with simple NLP features.

    Args:
        movies: A list of movie dictionaries.

    Returns:
        The same movies with ``sentiment`` and ``keywords`` added.
    """
    enriched: List[Dict] = []
    for movie in movies:
        overview = movie.get("overview", "") or ""
        tokens = normalize_text(overview).split()

        positive_words = {"love", "joy", "hope", "beautiful", "adventure", "magic", "dream", "friend", "family", "fun", "war"}
        negative_words = {"dark", "death", "murder", "violence", "tragedy", "sad", "evil", "nightmare", "loss", "fear"}

        positive_hits = sum(1 for token in tokens if token in positive_words)
        negative_hits = sum(1 for token in tokens if token in negative_words)
        sentiment = 0.0
        if positive_hits or negative_hits:
            sentiment = (positive_hits - negative_hits) / max(positive_hits + negative_hits, 1)

        movie_copy = dict(movie)
        movie_copy["sentiment"] = round(sentiment, 4)
        movie_copy["keywords"] = [token for token in tokens if len(token) > 3][:10]
        enriched.append(movie_copy)
    return enriched
