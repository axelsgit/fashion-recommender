# hybrid_recommender.py
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from collaborative_recommender import build_user_item_matrix, train_item_similarity_model, recommend_items
from content_recommender import load_items, build_item_profiles, recommend_for_user
from visual_recommender import load_features, recommend_similar_images


# --- Utility functions ---

def normalize_series(series: pd.Series) -> pd.Series:
    """Min-max normalize a pandas Series to [0,1]."""
    if series.empty:
        return series
    scaler = MinMaxScaler()
    values = scaler.fit_transform(series.values.reshape(-1, 1)).flatten()
    return pd.Series(values, index=series.index)

def compute_alpha(user_id, user_item_matrix, min_alpha=0.2, max_alpha=0.6):
    """Adaptive alpha: more interactions -> higher alpha (collab weight)."""
    if user_id not in user_item_matrix.index:
        return min_alpha
    interaction_count = (user_item_matrix.loc[user_id] > 0).sum()
    # Cap normalization at 50 interactions
    norm = min(interaction_count / 50, 1.0)
    return min_alpha + (max_alpha - min_alpha) * norm

# --- Diversification (MMR) ---

def diversify_mmr(item_scores, item_similarity, top_k=10, lambda_param=0.7):
    """Diversify recommendations using Maximal Marginal Relevance (MMR)."""
    selected = []
    candidates = list(item_scores.index)

    while len(selected) < top_k and candidates:
        mmr_scores = []
        for item in candidates:
            relevance = item_scores[item]
            diversity = max([item_similarity.loc[item, s] for s in selected], default=0)
            mmr_score = lambda_param * relevance - (1 - lambda_param) * diversity
            mmr_scores.append((item, mmr_score))

        # pick best candidate
        best_item = max(mmr_scores, key=lambda x: x[1])[0]
        selected.append(best_item)
        candidates.remove(best_item)

    return pd.Series(item_scores.loc[selected])

# --- Hybrid recommender ---

def hybrid_recommend(user_id, user_item_matrix, item_similarity_collab,
                     df_items, item_similarity_content, df_interactions, df_users,
                     features=None, img_paths=None,
                     query_image=None,
                     alpha=None, beta=None, gamma=None,
                     top_k=50, diversify=False, lambda_param=0.7):
    """
    Combine collaborative, content-based, and visual recommendations.
    - features/img_paths: precomputed ResNet embeddings (for visual recommender).
    - query_image: path of image used as a visual query.
    - alpha, beta, gamma: weights (if None, adaptive with defaults).
    """

    # --- Collaborative ---
    try:
        collab_scores = recommend_items(
            user_id, user_item_matrix, item_similarity_collab,
            df_interactions, top_k=top_k*5
        )
    except KeyError:
        collab_scores = pd.Series(dtype=float)

    # --- Content ---
    try:
        content_scores = recommend_for_user(
            user_id, df_items, item_similarity_content,
            df_interactions, df_users, top_k=top_k*5
        )
    except ValueError:
        content_scores = pd.Series(dtype=float)

    # --- Visual (only if query_image given) ---
    if features is not None and img_paths is not None and query_image:
        visual_raw = recommend_similar_images(query_image, features, img_paths, top_k=top_k*5)
        visual_scores = pd.Series(
            {img: score for img, score in visual_raw}, dtype=float
        )
    else:
        visual_scores = pd.Series(dtype=float)

    # Normalize
    collab_scores = normalize_series(collab_scores.groupby(collab_scores.index).max())
    content_scores = normalize_series(content_scores.groupby(content_scores.index).max())
    visual_scores = normalize_series(visual_scores.groupby(visual_scores.index).max())

    # Decide weights
    if alpha is None:
        alpha = compute_alpha(user_id, user_item_matrix)
    if beta is None or gamma is None:
        leftover = 1 - alpha
        beta = leftover * 0.67  # favor content slightly
        gamma = leftover * 0.33 # visual weaker by default

    # Merge all
    all_scores = pd.concat([collab_scores, content_scores, visual_scores], axis=1).fillna(0)
    all_scores.columns = ["collab", "content", "visual"]
    all_scores["final"] = (
        alpha * all_scores["collab"] +
        beta * all_scores["content"] +
        gamma * all_scores["visual"]
    )

    # Filter to items in catalog
    valid_items = set(df_items["image_path"].unique())
    all_scores = all_scores[all_scores.index.isin(valid_items)]

    final_scores = all_scores["final"].sort_values(ascending=False)

    # Diversification
    if diversify:
        final_scores = diversify_mmr(final_scores, item_similarity_content,
                                     top_k=top_k, lambda_param=lambda_param)
    else:
        final_scores = final_scores.head(top_k)

    # Fallback
    if len(final_scores) < top_k:
        missing = top_k - len(final_scores)
        popular_items = (df_interactions["item_id"]
                         .value_counts()
                         .index.difference(final_scores.index))[:missing]
        popular_scores = pd.Series([0.01]*len(popular_items), index=popular_items)
        final_scores = pd.concat([final_scores, popular_scores])

    return final_scores.head(top_k)
