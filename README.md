# ♻️ Waste Classification Demo

An interactive web app that classifies waste images into **30 categories**
using a deep learning model built with EfficientNetB0.

🔗 **Live Demo:** https://waste-classification-demo-fwv65knoqoz7dgfwdsauhn.streamlit.app
🤗 **Model on Hugging Face:** https://huggingface.co/Darshan764/waste-classification-v2
---

## 🎯 What It Does

Upload a photo of any waste item — a bottle, a can, a newspaper, food scraps —
and the app tells you which of 30 waste categories it belongs to, with
confidence scores for the top 3 predictions.

## 🧠 Model

- **Architecture:** EfficientNetB0 + custom classification head
- **Input size:** 224 × 224 × 3
- **Training data:** 15,000 images across 30 classes
  ([Recyclable and Household Waste Classification](https://www.kaggle.com/datasets/alistairking/recyclable-and-household-waste-classification))
- **Validation accuracy:** 87.3% on 3,000 held-out images
- **Framework:** TensorFlow / Keras

The model is loaded directly from the Hugging Face Hub at runtime —
no weights are stored in this repository.

## 📦 The 30 Classes

`aerosol_cans`, `aluminum_food_cans`, `aluminum_soda_cans`,
`cardboard_boxes`, `cardboard_packaging`, `clothing`, `coffee_grounds`,
`disposable_plastic_cutlery`, `eggshells`, `food_waste`,
`glass_beverage_bottles`, `glass_cosmetic_containers`, `glass_food_jars`,
`magazines`, `newspaper`, `office_paper`, `paper_cups`, `plastic_cup_lids`,
`plastic_detergent_bottles`, `plastic_food_containers`,
`plastic_shopping_bags`, `plastic_soda_bottles`, `plastic_straws`,
`plastic_trash_bags`, `plastic_water_bottles`, `shoes`, `steel_food_cans`,
`styrofoam_cups`, `styrofoam_food_containers`, `tea_bags`

## 🚀 Run Locally

```bash
# Clone the repository
git clone https://github.com/Darshan764/waste-classification-demo.git
cd waste-classification-demo

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
