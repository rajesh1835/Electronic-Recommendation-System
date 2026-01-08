def clean_data(df):
    """
    Clean raw dataset
    """
    df = df.dropna()

    df['category'] = df['category'].str.lower()
    df['brand'] = df['brand'].str.lower()

    df = df[df['price'] > 0]
    df = df[df['rating'] > 0]

    return df