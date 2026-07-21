from recommender import HybridRecommender


def test_recommendations_include_related_titles():
    movies = [
        {
            "id": 1,
            "title": "The Matrix",
            "overview": "A hacker discovers the truth about reality and joins a rebellion against machines.",
            "genres": ["Sci-Fi", "Action"],
            "vote_average": 8.7,
            "popularity": 82.4,
            "release_date": "1999-03-31",
            "poster_path": "/matrix.jpg",
            "sentiment": 0.1,
        },
        {
            "id": 2,
            "title": "The Matrix Reloaded",
            "overview": "The resistance continues its fight as the machines tighten their grip on reality.",
            "genres": ["Sci-Fi", "Action"],
            "vote_average": 7.2,
            "popularity": 74.1,
            "release_date": "2003-05-15",
            "poster_path": "/matrix2.jpg",
            "sentiment": 0.0,
        },
        {
            "id": 3,
            "title": "Inception",
            "overview": "A thief enters dreams to steal secrets and must plant an idea in a target's mind.",
            "genres": ["Sci-Fi", "Thriller"],
            "vote_average": 8.8,
            "popularity": 88.0,
            "release_date": "2010-07-16",
            "poster_path": "/inception.jpg",
            "sentiment": 0.05,
        },
        {
            "id": 4,
            "title": "The Grand Budapest Hotel",
            "overview": "A concierge and his protégé navigate a changing Europe with wit and charm.",
            "genres": ["Comedy", "Drama"],
            "vote_average": 8.4,
            "popularity": 52.3,
            "release_date": "2014-02-26",
            "poster_path": "/budapest.jpg",
            "sentiment": 0.7,
        },
    ]

    recommender = HybridRecommender(movies)
    results = recommender.recommend("The Matrix", top_n=3)

    assert len(results) == 3
    assert any(item["title"] == "The Matrix Reloaded" for item in results)
    assert all(item["title"] != "The Matrix" for item in results)
