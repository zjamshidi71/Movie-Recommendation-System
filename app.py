"""Streamlit app entry point for the movie recommender.

This app provides a polished UI around the hybrid recommender and TMDB-backed data layer.
It is designed to be deployable on Hugging Face Spaces with minimal configuration.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

import streamlit as st

API_KEY = os.getenv("TMDB_API_KEY", "")

from data_fetch import TMDBClient
from nlp_processing import extract_features
from recommender import HybridRecommender


st.set_page_config(page_title="Plexora Recommender", page_icon="🎬", layout="wide")


@st.cache_resource
def load_movies() -> List[dict]:
    """Load movie data once per session and enrich it with NLP features."""
    client = TMDBClient(api_key=API_KEY)
    movies = client.fetch_popular_movies(limit=80)
    if not movies:
        return [
            {
                "id": 1,
                "title": "The Matrix",
                "overview": "A hacker discovers the truth about reality and joins a rebellion against machines.",
                "genres": ["Sci-Fi", "Action"],
                "vote_average": 8.7,
                "popularity": 82.4,
                "release_date": "1999-03-31",
                "poster_path": "",
                "director": "Lana Wachowski",
                "cast": ["Keanu Reeves", "Laurence Fishburne"],
                "sentiment": 0.1,
                "revenue": 463517383,
                "budget": 63000000,
                "runtime": 136,
            }
        ]
    return extract_features(movies)


@st.cache_data
def build_recommender() -> HybridRecommender:
    """Build a reusable recommender instance for the current movie catalog."""
    return HybridRecommender(load_movies())


def poster_url(path: str) -> str:
    """Construct a full poster URL from a TMDB poster path."""
    if not path:
        return "https://via.placeholder.com/300x450?text=No+Poster"
    return f"https://image.tmdb.org/t/p/w500{path}"


def render_movie_card(movie: dict) -> None:
    """Render a single movie recommendation card in the UI."""
    with st.container():
        col1, col2 = st.columns([1, 3])
        with col1:
            st.image(poster_url(movie.get("poster_path", "")), use_container_width=True)
        with col2:
            st.subheader(movie.get("title", "Untitled"))
            st.caption(f"Release: {movie.get('release_date') or 'Unknown'} | Rating: {movie.get('vote_average', 'N/A')}")
            st.write(movie.get("overview", "No description available."))
            st.write(f"Genres: {', '.join(movie.get('genres', [])) or 'Unknown'}")
            st.write(f"Director: {movie.get('director') or 'Unknown'}")
            st.write(f"Cast: {', '.join(movie.get('cast', []))[:120] or 'Unknown'}")
            if movie.get("revenue"):
                st.write(f"Revenue: ${movie['revenue']:,}")
            st.write(f"Reason: {movie.get('reason', 'Similar themes and strong audience appeal.')}")


def main() -> None:
    """Render the full application shell."""
    st.title("Plexora — AI Movie Recommender")
    st.caption("Discover films that blend similar themes, audience appeal, and mood.")

    with st.sidebar:
        st.image("https://via.placeholder.com/220x80?text=Plexora", width=220)
        st.header("Filters")
        if not API_KEY:
            st.warning("TMDB_API_KEY is not set in this session, so the app is using fallback sample data.")
        selected_title = st.selectbox("Choose a movie", [movie["title"] for movie in load_movies()], index=0)
        mood = st.selectbox("Mood", ["Any", "Light", "Dark", "Inspiring", "Thrilling"], index=0)
        top_n = st.slider("Number of recommendations", min_value=3, max_value=10, value=6)
        st.markdown("""### Setup
- TMDB is configured for this environment using the provided key.
- For future deployments, store the same key in Hugging Face Space secrets or a local environment variable.
- If the API is unavailable, fallback sample data is used.
""")

    movies = load_movies()
    recommender = build_recommender()
    results = recommender.recommend(selected_title, top_n=top_n)

    if mood != "Any":
        results = [movie for movie in results if movie.get("sentiment", 0.0) >= 0.0]

    for index, movie in enumerate(results, start=1):
        base_reason = (
            "similar genre overlap and description themes" if movie.get("score", 0.0) > 0.6 else "strong audience appeal and related mood"
        )
        movie["reason"] = f"Recommended because it shares {base_reason} with {selected_title}."
        render_movie_card(movie)
        if index < len(results):
            st.divider()

    if not results:
        st.info("No recommendations were found. Try another title.")


if __name__ == "__main__":
    main()
