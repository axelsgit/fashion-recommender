# content_recommender.py
# recommend items that are similar in attributes (category, brand, description, etc.) to items a user liked

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

test_sample = 2000  # adjust based on memory and speed requirements


def load_items(sample_size, random_state=42):
    """
    Loads item metadata and optionally samples a subset for faster training/testing.
    Adjust sample_size depending on available resources.
    """
    df_items = pd.read_csv("products.csv")

    # If sample_size is smaller than dataset, take a random sample
    if sample_size and sample_size < len(df_items):
        df_items = df_items.sample(n=sample_size, random_state=random_state).reset_index(drop=True)
        print(f"Sampled {sample_size} items out of {len(df_items)} total.")
    else:
        print(f"Loaded full dataset with {len(df_items)} items.")

    return df_items


def build_item_profiles(df_items):
    """
    Build TF-IDF matrix from product text attributes.
    """

    # Identify attribute columns (exclude known metadata columns)
    meta_cols = ["image_path", "brand", "category_name", "description", "collection", "price", "text"]
    attr_cols = [col for col in df_items.columns if col not in meta_cols]

    # For each product, join the names of attributes where value == 1
    df_items["attr_text"] = df_items[attr_cols].apply(
        lambda row: " ".join([col for col in attr_cols if row[col] == 1]), axis=1
    )

    # Create a combined text field
    # TF-IDF transforms product text into numbers -> compute similarity -> to recommend fashion items based on content
    df_items["text"] = (
        df_items["category_name"].fillna("") + " " +
        df_items["brand"].fillna("") + " " +
        df_items["description"].fillna("") + " " +
        df_items["collection"].fillna("") + " " +
        df_items["price"].astype(str).fillna("") + " " +
        df_items["attr_text"].fillna("")
    )

    # Basic text cleaning
    df_items["text"] = df_items["text"].str.lower().str.strip()

    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(max_features=10000, stop_words="english") # max_features to limit runtime
    tfidf_matrix = vectorizer.fit_transform(df_items["text"])

    # Item-to-item similarity matrix
    similarity = linear_kernel(tfidf_matrix, tfidf_matrix)
    sim_df = pd.DataFrame(similarity, index=df_items["image_path"], columns=df_items["image_path"])

    return sim_df


def recommend_for_user(user_id, df_items, item_similarity, df_interactions, df_users, top_k=50):
    # Cold-start: new user
    if user_id not in df_interactions["user_id"].unique():
        # recommend popular or random catalog items
        fallback = df_items.sample(top_k, random_state=42)
        return pd.Series([1.0]*len(fallback), index=fallback["image_path"])

    user_meta = df_users[df_users["user_id"] == user_id].iloc[0].to_dict()
    style_pref = user_meta["style_pref"]

    user_items = df_interactions[df_interactions["user_id"] == user_id].sort_values("timestamp", ascending=False)
    recent_items = user_items["image_path"].head(5).tolist()

    scores = pd.Series(0, index=item_similarity.index, dtype=float)
    for rank, item in enumerate(recent_items):
        if item in item_similarity.columns:
            weight = 1.0 / (rank + 1)
            scores = scores.add(item_similarity[item] * weight, fill_value=0)

    style_mask = df_items.set_index("image_path")["description"].str.contains(style_pref, case=False, na=False)
    scores.loc[style_mask[style_mask].index] *= 1.2

    seen_items = user_items["image_path"].unique()
    scores = scores.drop(seen_items, errors="ignore")

    return scores.sort_values(ascending=False).head(top_k)


def recommend_similar_items(item_id, item_similarity, top_k=10):
    """
    Recommend top_k items that are most similar to a given item.
    Uses content-based similarity.
    """
    if item_id not in item_similarity.index:
        raise ValueError(f"Item {item_id} not found in similarity matrix.")

    scores = item_similarity[item_id].sort_values(ascending=False)
    scores = scores.drop(item_id, errors="ignore")  # remove itself
    return scores.head(top_k)