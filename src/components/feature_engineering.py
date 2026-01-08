from sklearn.preprocessing import MinMaxScaler

def create_features(df):
    """
    Create features for content-based recommendation
    """

    # Text feature for content-based filtering
    df['text_feature'] = df['category'] + " " + df['brand']

    # Normalize numeric features
    scaler = MinMaxScaler()

    df[['rating_norm', 'purchase_norm']] = scaler.fit_transform(
        df[['rating', 'purchase_count']]
    )

    # Value for money
    df['value_for_money'] = df['rating'] / df['price']
    df[['value_norm']] = scaler.fit_transform(df[['value_for_money']])

    return df
