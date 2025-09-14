# visual_recommender.py
import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from numpy.linalg import norm
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from keras.layers import GlobalMaxPooling2D
import tensorflow as tf


# Run a one-time indexing (this builds features.pkl + imagefiles.pkl):
# python -c "from visual_recommender import build_feature_index_from_catalog; build_feature_index_from_catalog('products.csv')"


# Load ResNet50 model 
base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224,224,3))
base_model.trainable = False
model = tf.keras.Sequential([
    base_model,
    GlobalMaxPooling2D()
])

# Feature extractor 
def extract_feature(img_path):
    img = image.load_img(img_path, target_size=(224,224,3))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img, verbose=0).flatten()
    return result / norm(result)

# Build feature index (save features + paths)
def build_feature_index_from_catalog(products_csv="products.csv", features_path="features.pkl", paths_path="imagefiles.pkl"):
    df_products = pd.read_csv(products_csv)
    catalog_image_paths = set(df_products["image_path"])

    #print(catalog_image_paths)

    valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    filenames = []

    # Walk through all subdirectories and collect only images in the catalog
    for root, dirs, files in os.walk("img"):
        for file in files:
            if file.lower().endswith(valid_exts):
                # Build relative path to match products.csv format
                rel_path = os.path.relpath(os.path.join(root, file))
                rel_path = rel_path.replace("\\", "/")
                if rel_path in catalog_image_paths:
                    filenames.append(os.path.join(root, file))

    feature_list = []
    #print(filenames[:5])
    for file in tqdm(filenames, desc="Extracting features"):
        try:
            feature = extract_feature(file)
            feature_list.append(feature)
        except Exception as e:
            print(f"Skipping {file}: {e}")

    pickle.dump(feature_list, open(features_path, "wb"))
    pickle.dump(filenames, open(paths_path, "wb"))
    print(f"Saved {len(feature_list)} features to {features_path} and {len(filenames)} paths to {paths_path}")

# Load precomputed features
def load_features(features_path="features.pkl", paths_path="imagefiles.pkl"):
    feature_list = pickle.load(open(features_path, "rb"))
    filenames = pickle.load(open(paths_path, "rb"))
    return feature_list, filenames

# Find similar images
def recommend_similar_images(query_img_path, feature_list, filenames, top_k=10):
    query_vector = extract_feature(query_img_path)
    similarities = [np.dot(query_vector, feat) for feat in feature_list]

    # Get indices sorted by similarity
    indices = np.argsort(similarities)[::-1]

    # Exclude the query image itself (exact path match)
    filtered = [(filenames[i], similarities[i]) for i in indices if os.path.abspath(filenames[i]) != os.path.abspath(query_img_path)]

    # Take top_k from the filtered list
    return filtered[:top_k]

