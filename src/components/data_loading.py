import pandas as pd

def load_data(path):
    """
    Load raw product dataset
    """
    df = pd.read_csv(path)
    return df
