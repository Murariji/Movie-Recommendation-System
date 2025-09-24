import pandas as pd
from load_data import preprocess
from sklearn.metrics.pairwise import cosine_similarity

class ItemCFRecommender:
    def __init__(self):
        self.movie_user_matrix = None
        self.similarity = None

    def fit(self, df):
        self.movie_user_matrix = df.pivot_table(index='title', columns='user_id', values='rating').fillna(0)
        self.similarity = pd.DataFrame(
            cosine_similarity(self.movie_user_matrix),
            index=self.movie_user_matrix.index,
            columns=self.movie_user_matrix.index
        )

    def recommend(self, user_ratings, top_n=10):
        scores = pd.Series(0, index=self.movie_user_matrix.index)
        for movie, rating in user_ratings.items():
            if movie in self.similarity.index:
                scores += self.similarity[movie] * rating

        for movie in user_ratings.keys():
            if movie in scores.index:
                scores.drop(movie, inplace=True)
        return scores.sort_values(ascending=False).head(top_n)

if __name__ == "__main__":
    df = preprocess()
    rec = ItemCFRecommender()
    rec.fit(df)
    
    user_ratings = {"Star Wars (1977)": 5, "Toy Story (1995)": 4}
    print("Top 10 recommendations for user:")
    print(rec.recommend(user_ratings, 10))
