# explain_recommendation.py

def explain_recommendation(item_id, user_id, df_items, df_users, df_interactions):
    # --- Safety checks ---
    if item_id not in df_items["image_path"].values:
        return "Recommended based on your general preferences (item not found in catalog)."
    if user_id not in df_users["user_id"].values:
        return "Recommended based on popular trends (new user with no profile)."

    user_meta = df_users[df_users["user_id"] == user_id].iloc[0]
    user_items = df_interactions[df_interactions["user_id"] == user_id]["image_path"].tolist()
    item_row = df_items[df_items["image_path"] == item_id].iloc[0]

    user_brands = df_items[df_items["image_path"].isin(user_items)]["brand"].values
    user_cats = df_items[df_items["image_path"].isin(user_items)]["category_name"].values

    # Hybrid, collaborative, content explanations
    reasons = []
    if item_row["category_name"] in user_cats:
        reasons.append(f"you viewed {item_row['category_name']} before")
    if item_row["brand"] in user_brands:
        reasons.append(f"you liked {item_row['brand']}")

    if reasons:
        return "Recommended because " + " and ".join(reasons) + "."
    else:
        return f"Recommended because it matches your style preference: {user_meta['style_pref']}."