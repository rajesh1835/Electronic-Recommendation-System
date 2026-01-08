from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd


class SVDRecommender:
    def __init__(self, df, n_components=50):
        self.df = df.copy()

        # TF-IDF
        self.vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = self.vectorizer.fit_transform(self.df["text_feature"])

        # Adjust n_components safely
        max_components = tfidf_matrix.shape[1] - 1
        self.n_components = min(n_components, max_components)

        # SVD
        self.svd = TruncatedSVD(
            n_components=self.n_components,
            random_state=42
        )
        self.latent_matrix = self.svd.fit_transform(tfidf_matrix)

    def recommend(self, query, top_n=10):
        """
        Recommend products using TF-IDF + SVD + ranking
        """

        # Transform query
        query_vec = self.vectorizer.transform([query.lower()])
        query_latent = self.svd.transform(query_vec)

        # Similarity
        similarity_scores = cosine_similarity(
            query_latent, self.latent_matrix
        ).flatten()

        # Final ranking score
        self.df["final_score"] = (
            0.5 * similarity_scores +
            0.2 * self.df["rating_norm"] +
            0.2 * self.df["purchase_norm"] +
            0.1 * self.df["value_norm"]
        )

        return (
            self.df
            .sort_values(by="final_score", ascending=False)
            .head(top_n)
            [["product_id", "category", "brand", "price", "rating", "final_score"]]
        )


