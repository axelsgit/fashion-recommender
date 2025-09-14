# collaborative_recommender.py
# user–item interaction matrix + item–item similarity

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def load_all_data():
    df_interactions = pd.read_csv("user_interactions.csv")
    return df_interactions


def build_user_item_matrix(df_interactions):
    matrix = df_interactions.pivot_table(
        index="user_id",
        columns="image_path",
        values="interaction_score",
        aggfunc="max",
        fill_value=0
    )
    return matrix


def train_item_similarity_model(user_item_matrix):
    similarity = cosine_similarity(user_item_matrix.T)  # item-to-item
    sim_df = pd.DataFrame(similarity,
                          index=user_item_matrix.columns,
                          columns=user_item_matrix.columns)
    return sim_df


def recommend_items(user_id, user_item_matrix, item_similarity, df_interactions, top_k=50):
    # Cold-start: New user
    if user_id not in user_item_matrix.index or user_item_matrix.loc[user_id].sum() == 0:
        if df_interactions is not None:
            popular_items = (
                df_interactions["image_path"]
                .value_counts()
                .head(top_k)
                .index.tolist()
            )
            return pd.Series([1.0]*len(popular_items), index=popular_items)
        else:
            return pd.Series(dtype=float)

    # Normal case
    user_vector = user_item_matrix.loc[user_id]
    scores = user_vector @ item_similarity
    already_seen = user_vector[user_vector > 0].index
    scores = scores.drop(already_seen, errors="ignore")
    return scores.sort_values(ascending=False).head(top_k)



