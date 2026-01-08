from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class TFIDFRecommender:
    def __init__(self, df):
        self.df = df.copy()
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.vectorizer.fit_transform(df['text_feature'])

    def recommend(self, query, top_n=10):
        """
        Recommend products based on search query
        """
        query_vec = self.vectorizer.transform([query.lower()])
        similarity_scores = cosine_similarity(
            query_vec, self.tfidf_matrix
        ).flatten()

        self.df['similarity_score'] = similarity_scores

        return (
            self.df
            .sort_values(by='similarity_score', ascending=False)
            .head(top_n)
            [['product_id', 'category', 'brand', 'price', 'rating', 'similarity_score']]
        )


