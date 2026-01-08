import pandas as pd

from src.models.popularity_based import PopularityRecommender
from src.models.tf_idf_content_based import TFIDFRecommender
from src.models.SVD import SVDRecommender


def main():
    # --------------------------------------------------
    # Load feature-engineered dataset
    # --------------------------------------------------
    df = pd.read_csv("data/processed/featured_products.csv")

    print("\n==============================================")
    print(" ELECTRONICS PRODUCT RECOMMENDATION SYSTEM")
    print("==============================================\n")

    # --------------------------------------------------
    # User search query
    # --------------------------------------------------
    query = "samsung mobile"
    top_n = 5

    # --------------------------------------------------
    # Model 1: Popularity-Based Recommendation
    # --------------------------------------------------
    print("=== BASELINE MODEL 1: POPULARITY-BASED ===\n")
    popularity_model = PopularityRecommender(df)
    print(popularity_model.recommend(top_n=top_n))

    # --------------------------------------------------
    # Model 2: TF-IDF Content-Based Recommendation
    # --------------------------------------------------
    print("\n=== BASELINE MODEL 2: TF-IDF CONTENT-BASED ===\n")
    tfidf_model = TFIDFRecommender(df)
    print(tfidf_model.recommend(query=query, top_n=top_n))

    # --------------------------------------------------
    # Model 3: TF-IDF + SVD + Ranking (Best Model)
    # --------------------------------------------------
    print("\n=== FINAL MODEL: TF-IDF + SVD + RANKING ===\n")
    svd_model = SVDRecommender(df)
    print(svd_model.recommend(query=query, top_n=top_n))

    print("\n==============================================")
    print(" Recommendation completed successfully ")
    print("==============================================\n")


if __name__ == "__main__":
    main()
