import pandas as pd
import random
from datetime import datetime, timedelta


path_to_categories_img = r"Category and Attribute Prediction Benchmark\Anno_coarse\list_category_img.txt"
path_to_categories_names = r"Category and Attribute Prediction Benchmark\Anno_coarse\list_category_cloth.txt"

path_to_attributes_img = r"Category and Attribute Prediction Benchmark\Anno_coarse\list_attr_img.txt"
path_to_attributes_names = r"Category and Attribute Prediction Benchmark\Anno_coarse\list_attr_cloth.txt"

# 1. Load categories and attributes

######### categories ###########
df_cat = pd.read_csv(path_to_categories_img,
                     sep='\s+', header=None, skiprows=2)

df_cat.columns = ["image_path", "category_id"] 

df_cat_names = pd.read_csv(path_to_categories_names, 
                     sep='\s+', header=None, skiprows=2)

df_cat_names.columns = ["category_id", "category_name", "category_label"]
df_cat_full = df_cat.merge(df_cat_names, on="category_id")

#print(df_cat_full.columns)
print(f"Loaded {len(df_cat_full)} items with categories")
######### categories ###########

######### attributes ###########
attr_names = []

with open(path_to_attributes_names, "r") as f:
    lines = f.readlines()

# Skip the first two lines (count + header)
for line in lines[2:]:
    parts = line.strip().split()
    if len(parts) >= 2:
        attr_name = " ".join(parts[:-1])
        attr_names.append(attr_name)

#print(f"Loaded {len(attr_names)} attributes")
#print(attr_names[:20])

# 2. Load image -> attribute matrix
df_attr = pd.read_csv(path_to_attributes_img, sep="\s+", skiprows=2, header=None)

# 3. Assign column names (image_path + 1000 attrs)
df_attr.columns = ["image_path"] + attr_names

print(f"Loaded attributes for {len(df_attr)} items")
######### attributes ###########

# Now merge with attributes
df = df_cat_full.merge(df_attr, on="image_path", how="inner")

# Drop duplicate image_path values BEFORE generating interactions
df = df.drop_duplicates(subset="image_path").reset_index(drop=True)

# --- REDUCE PRODUCT CATALOG FOR EXPERIMENTATION ---
# Sample 10,000 products for faster experimentation
df = df.sample(n=10000, random_state=42).reset_index(drop=True)

#print(df.columns)

brands = [
    # Japanese
    "Comme des Garcons", "Issey Miyake", "Yohji Yamamoto",
    # English
    "Burberry", "Alexander McQueen",
    # French
    "Chanel", "Louis Vuitton", "Dior",
    # Italian
    "Gucci", "Prada"
]

descriptions = [
    "An exclusive luxury design crafted with precision.",
    "A timeless piece from a world-renowned fashion house.",
    "Handmade using the finest materials for ultimate quality.",
    "Iconic style that embodies elegance and sophistication.",
    "Limited edition release representing high-end fashion."
]

collections = [
    "Spring/Summer 2024", "Fall/Winter 2024",
    "Cruise 2025", "Resort 2025",
    "Spring/Summer 2025", "Fall/Winter 2025",
    "Haute Couture 2024", "Pre-Fall 2025"
]

df["price"] = df.apply(lambda x: round(random.uniform(500, 5000), 2), axis=1)  
df["brand"] = df.apply(lambda x: random.choice(brands), axis=1)
df["description"] = df.apply(lambda x: random.choice(descriptions), axis=1)
df["collection"] = df.apply(lambda x: random.choice(collections), axis=1)

#print(df.columns)


# 4. Create User Interactions for later training/testing of recommender

interaction_weights = {
    "view": 1,
    "wishlist": 2,
    "cart": 3,
    "purchase": 5
}

num_users = 500
user_ids = [f"user_{i}" for i in range(1, num_users+1)]

# Simulate interactions
interactions = []
for user in user_ids:
    # Each user interacts with 100-200 random items
    n_items = random.randint(100, 200)
    sampled_items = df.sample(n_items).to_dict(orient="records")
    
    for item in sampled_items: # assign interaction type
        interaction_type = random.choices(
            list(interaction_weights.keys()),
            weights=[0.6, 0.15, 0.15, 0.1],  
            # more likely to "view" than "purchase" 
            # -> probability distribution: 60% chance → "view", 15% chance → "wishlist", 15% chance → "cart", 10% chance → "purchase"
            k=1
        )[0]
        
        interactions.append({
            "user_id": user,
            "image_path": item["image_path"],
            "brand": item["brand"],
            "category_name": item["category_name"],
            "interaction": interaction_type,
            "interaction_score": interaction_weights[interaction_type]
        })

# Build dataframe
df_interactions = pd.DataFrame(interactions)

# Save for later use
df_interactions.to_csv("user_interactions.csv", index=False)

# Save product catalog for content-based recommender
product_cols = ["image_path", "brand", "category_name", "description", "collection", "price"] + attr_names
df_products = df[product_cols].drop_duplicates()
# Reduce the number of items to half
df_products = df_products.sample(frac=0.5, random_state=42).reset_index(drop=True)
df_products.to_csv("products.csv", index=False)
print("Saved products.csv with", df_products.shape[0], "items.")

print("Generated relevant data files: user_interactions.csv, products.csv")

# --- Generate user metadata ---
user_profiles = []
for user in user_ids:
    age = random.randint(18, 60)
    gender = random.choice(["male", "female", "unisex"])
    location = random.choice(["Munich", "Paris", "London", "Berlin"])
    style_pref = random.choice(["casual", "luxury", "streetwear", "minimalist"])

    user_profiles.append({
        "user_id": user,
        "age": age,
        "gender": gender,
        "location": location,
        "style_pref": style_pref
    })

df_users = pd.DataFrame(user_profiles)
df_users.to_csv("users.csv", index=False)
print("Saved users.csv with user metadata")

# --- Add timestamps to interactions ---
for i in range(len(interactions)):
    days_ago = random.randint(0, 180)  # last 6 months
    timestamp = datetime.now() - timedelta(days=days_ago)
    interactions[i]["timestamp"] = timestamp.strftime("%Y-%m-%d %H:%M:%S")

df_interactions = pd.DataFrame(interactions)
df_interactions.to_csv("user_interactions.csv", index=False)
