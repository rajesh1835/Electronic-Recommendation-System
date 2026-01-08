import pandas as pd

class PopularityRecommender:
    def __init__(self, df):
        self.df = df.copy()

    def recommend(self, top_n=10):
        """
        Recommend top-N most popular electronics products
        """
        return (
            self.df
            .sort_values(by='purchase_count', ascending=False)
            .head(top_n)
            [['product_id', 'category', 'brand', 'price', 'rating', 'purchase_count']]
        )

