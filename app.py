# app.py
# python -m uvicorn app:app --reload

from fastapi import FastAPI
import pandas as pd
from collaborative_recommender import build_user_item_matrix, train_item_similarity_model, recommend_items
from content_recommender import load_items, build_item_profiles, recommend_for_user
from hybrid_recommender import hybrid_recommend

# Load data & models at startup
df_interactions = pd.read_csv("user_interactions.csv")
df_items = pd.read_csv("products.csv")
user_item_matrix = build_user_item_matrix(df_interactions)
item_similarity_collab = train_item_similarity_model(user_item_matrix)
item_similarity_content = build_item_profiles(df_items)

app = FastAPI()

@app.get("/recommend/user/{user_id}")
def recommend_for_user_api(user_id: str, top_k: int = 10):
    recs = hybrid_recommend(
        user_id, user_item_matrix, item_similarity_collab,
        df_items, item_similarity_content, df_interactions, df_users=None,
        top_k=top_k
    )
    return {"user_id": user_id, "recommendations": recs.head(top_k).to_dict()}

@app.get("/recommend/item/{item_id}")
def recommend_similar_item(item_id: str, top_k: int = 5):
    if item_id not in item_similarity_content.index:
        return {"error": "Item not found"}
    sims = item_similarity_content[item_id].sort_values(ascending=False).head(top_k+1).drop(item_id)
    return {"item_id": item_id, "similar_items": sims.to_dict()}
