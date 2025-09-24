import pandas as pd
from load_data import preprocess

class PopularityRecommender:
    def __init__(self):
        self.movie_scores = None

    def fit(self, df):
        movie_stats = df.groupby("title").agg({"rating": ["mean", "count"]})
        movie_stats.columns = ["avg_rating", "num_ratings"]
        self.movie_scores = movie_stats[movie_stats["num_ratings"] >= 50].sort_values(
            by=["avg_rating", "num_ratings"], ascending=False
        )

    def recommend(self, top_n=10):
        return self.movie_scores.head(top_n)

if __name__ == "__main__":
    df = preprocess()
    rec = PopularityRecommender()
    rec.fit(df)
    print("Top 10 popular movies:")
    print(rec.recommend(10))
