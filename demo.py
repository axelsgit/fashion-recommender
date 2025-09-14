# demo.py 
# Run with: streamlit run demo.py

import os
import base64
import streamlit as st
import pandas as pd
from hybrid_recommender import hybrid_recommend
from collaborative_recommender import build_user_item_matrix, train_item_similarity_model
from content_recommender import load_items, build_item_profiles
from explain_recommendation import explain_recommendation
from visual_recommender import load_features, recommend_similar_images

# Helper: Encode images as Base64 for HTML
def image_to_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# Load data & models
df_interactions = pd.read_csv("user_interactions.csv")
df_items = pd.read_csv("products.csv")
df_users = pd.read_csv("users.csv")

user_item_matrix = build_user_item_matrix(df_interactions)
item_similarity_collab = train_item_similarity_model(user_item_matrix)
item_similarity_content = build_item_profiles(df_items)

features, img_paths = load_features()

st.title("Fashion Recommender System")

# Mode Selection 
mode = st.radio("Choose Recommendation Mode", ["Hybrid", "Visual"])

top_k = st.slider("Number of Recommendations", 5, 20, 10)

# ============================
# HYBRID MODE
# ============================
if mode == "Hybrid":
    user_id = st.selectbox("Select a User", df_interactions["user_id"].unique())

    recs = hybrid_recommend(
        user_id, user_item_matrix, item_similarity_collab,
        df_items, item_similarity_content, df_interactions, df_users,
        top_k=top_k
    )

    st.subheader(f"Hybrid Recommendations for {user_id}:")
    cols = st.columns(2)

    for i, item in enumerate(recs.index):
        item_row = df_items[df_items["image_path"] == item].iloc[0]
        image_path = os.path.join(item)

        with cols[i % 2]:
            if os.path.exists(image_path):
                img_base64 = image_to_base64(image_path)
                img_tag = f"<img src='data:image/jpeg;base64,{img_base64}' class='rec-img'/>"
            else:
                img_tag = "<div style='color:red;'>Image not found</div>"

            explanation = explain_recommendation(item, user_id, df_items, df_users, df_interactions)

            card_html = f"""
            <div class="rec-card">
                {img_tag}
                <div class="rec-info">
                    <b>{item_row['brand']}</b> – {item_row['category_name']} – {item_row['price']} €
                    <br><span class="desc">{item_row['description']}</span>
                    <div class="caption">{explanation}</div>
                </div>
            </div>
            """

            st.markdown(card_html, unsafe_allow_html=True)

# ============================
# VISUAL MODE
# ============================
elif mode == "Visual":
    user_id = st.selectbox("Select a User for Visual Recommendations", df_interactions["user_id"].unique())
    user_items = df_interactions[df_interactions["user_id"] == user_id].sort_values("timestamp", ascending=False)

    if not user_items.empty:
        query_image_path = user_items.iloc[0]["image_path"]
        query_image_full_path = os.path.join(query_image_path)
        st.image(query_image_full_path, caption="Most Recent Interacted Item", width=250)

        recs = recommend_similar_images(query_image_full_path, features, img_paths, top_k=top_k)

        st.subheader("Visually Similar Items:")
        cols = st.columns(2)

        for i, (rec_path, score) in enumerate(recs):
            with cols[i % 2]:
                if os.path.exists(rec_path):
                    img_base64 = image_to_base64(rec_path)
                    img_tag = f"<img src='data:image/jpeg;base64,{img_base64}' class='rec-img'/>"
                else:
                    img_tag = "<div style='color:red;'>Image not found</div>"

                card_html = f"""
                <div class="rec-card">
                    {img_tag}
                </div>
                """
                st.markdown(card_html, unsafe_allow_html=True)
    else:
        st.info("This user has no interactions yet.")

# ============================
# CSS Styling
# ============================
st.markdown("""
<style>
.rec-card {
    border: 1px solid #ddd;
    border-radius: 10px;
    padding: 10px;
    margin-bottom: 20px;
    text-align: center;
    background-color: #fff;
    box-shadow: 0 2px 6px rgba(0,0,0,0.08);
    display: flex;
    flex-direction: column;
    justify-content: flex-start;
    height: 100%;  /* cards stretch equally */
}
.rec-img {
    max-width: 200px;
    max-height: 250px;
    object-fit: contain;  /* no cropping */
    margin: 0 auto 10px auto;
}
.rec-info {
    text-align: left;
    font-size: 0.9em;
    flex-grow: 1;  /* ensures text block grows and aligns */
}
.desc {
    color: gray;
    font-size: 0.85em;
}
.caption {
    color: #666;
    font-size: 0.8em;
    margin-top: 5px;
}
[data-testid="column"] > div {
    display: flex;
    flex-direction: column;
    height: 100%;  /* makes all columns in a row stretch equally */
}
</style>
""", unsafe_allow_html=True)

