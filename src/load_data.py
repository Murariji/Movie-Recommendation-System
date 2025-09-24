import pandas as pd
from pathlib import Path

# Paths
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "ml-100k"

def load_ratings():
    ratings_path = DATA_DIR / "u.data"
    cols = ["user_id", "movie_id", "rating", "timestamp"]
    ratings = pd.read_csv(ratings_path, sep="\t", names=cols)
    return ratings

def load_movies():
    
    movies_path = DATA_DIR / "u.item"
    cols = [
        "movie_id", "title", "release_date", "video_release_date", "IMDb_URL",
        "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
        "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery",
        "Romance", "Sci-Fi", "Thriller", "War", "Western"
    ]
    movies = pd.read_csv(movies_path, sep="|", names=cols, encoding="latin-1")
    return movies

def preprocess():
    ratings = load_ratings()
    movies = load_movies()
    data = pd.merge(ratings, movies, on="movie_id")
    return data

if __name__ == "__main__":
    df = preprocess()
    print("Dataset shape:", df.shape)
    print(df.head())
