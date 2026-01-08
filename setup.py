from setuptools import setup, find_packages

setup(
    name="search-based-product-recommender",
    version="1.0.0",
    author="Rajesh",
    description="Search-based content-based product recommendation system",
    long_description="A content-based recommendation system that suggests products based on search queries using TF-IDF and cosine similarity.",
    packages=find_packages(where="."),
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "streamlit",
        "matplotlib",
        "seaborn",
        "plotly",
        "flask"
    ],
    python_requires=">=3.8",
)
