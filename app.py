import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download

@st.cache_resource
def load_model():
    # Download the model file to Streamlit's local cache
    model_path = hf_hub_download(
        repo_id="Darshan764/waste-classification-v2",
        filename="best_waste_classifier.keras"  # Check the exact filename in your HF repo
    )
    # Now load using the local file path
    return tf.keras.models.load_model(model_path)

model = load_model()

# ... rest of your app.py stays the same

CLASS_NAMES = [
    "aerosol_cans", "aluminum_food_cans", "aluminum_soda_cans",
    "cardboard_boxes", "cardboard_packaging", "clothing",
    "coffee_grounds", "disposable_plastic_cutlery", "eggshells",
    "food_waste", "glass_beverage_bottles", "glass_cosmetic_containers",
    "glass_food_jars", "magazines", "newspaper", "office_paper",
    "paper_cups", "plastic_cup_lids", "plastic_detergent_bottles",
    "plastic_food_containers", "plastic_shopping_bags",
    "plastic_soda_bottles", "plastic_straws", "plastic_trash_bags",
    "plastic_water_bottles", "shoes", "steel_food_cans",
    "styrofoam_cups", "styrofoam_food_containers", "tea_bags"
]

st.title("♻️ Waste Classification V2")
st.write("Upload an image of waste to classify it into one of 30 categories.")

uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img = image.resize((224, 224))
    arr = np.expand_dims(np.array(img), axis=0)

    with st.spinner("Classifying..."):
        preds = model.predict(arr, verbose=0)[0]

    idx = int(np.argmax(preds))
    st.success(f"**Prediction:** {CLASS_NAMES[idx]}")
    st.metric("Confidence", f"{preds[idx]*100:.1f}%")

    st.subheader("Top 3 Predictions")
    top3 = np.argsort(preds)[-3:][::-1]
    for i in top3:
        st.progress(
            float(preds[i]),
            text=f"{CLASS_NAMES[i]}: {preds[i]*100:.1f}%"
        )
