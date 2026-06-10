import streamlit as st
import pandas as pd
import numpy as np
import pickle

# -----------------------------------
# Load Model
# -----------------------------------

with open("house_price_model.pkl", "rb") as f:
    model = pickle.load(f)

# -----------------------------------
# Load Dataset
# -----------------------------------

df = pd.read_csv("kc_house_data.csv")
df = df.drop(columns=["id", "date"])

# -----------------------------------
# Nearest House Function
# -----------------------------------

def get_nearest_house(df, bedrooms, bathrooms, sqft_living):

    temp = df.copy()

    temp["distance"] = (
        abs(temp["bedrooms"] - bedrooms)
        + abs(temp["bathrooms"] - bathrooms)
        + abs(temp["sqft_living"] - sqft_living) / 100
    )

    nearest_house = temp.loc[temp["distance"].idxmin()]

    return nearest_house


# -----------------------------------
# Page Config
# -----------------------------------

st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# -----------------------------------
# Title
# -----------------------------------

st.title("🏠 House Price Predictor")

st.markdown("""
Predict house prices using Machine Learning.

### Enter:
- 📍 Location (Zipcode)
- 🛏 Bedrooms
- 🚿 Bathrooms
- 📐 Sqft Living

The app automatically finds the most similar house in that location and predicts its price.
""")

# -----------------------------------
# Sidebar Inputs
# -----------------------------------

st.sidebar.header("🏡 Property Details")

zipcode = st.sidebar.selectbox(
    "📍 Select Zipcode",
    sorted(df["zipcode"].unique())
)

bedrooms = st.sidebar.slider(
    "🛏 Bedrooms",
    min_value=1,
    max_value=10,
    value=3
)

bathrooms = st.sidebar.slider(
    "🚿 Bathrooms",
    min_value=1,
    max_value=10,
    value=2,
    
)

sqft_living = st.sidebar.number_input(
    "📐 Sqft Living",
    min_value=300,
    value=2000,
    max_value=14000,
)

predict_button = st.sidebar.button(
    "🚀 Predict Price"
)

# -----------------------------------
# Prediction
# -----------------------------------

if predict_button:

    filtered_df = df[df["zipcode"] == zipcode]

    nearest = get_nearest_house(
        filtered_df,
        bedrooms,
        bathrooms,
        sqft_living
    )

    input_data = nearest.drop("price").to_frame().T

    input_data["bedrooms"] = bedrooms
    input_data["bathrooms"] = bathrooms
    input_data["sqft_living"] = sqft_living

    # Engineered Feature
    input_data["grade_area"] = (
        input_data["grade"] * sqft_living
    )

    # Correct Column Order
    input_data = input_data[
        [
            'bedrooms',
            'bathrooms',
            'sqft_living',
            'sqft_lot',
            'floors',
            'waterfront',
            'view',
            'condition',
            'grade',
            'sqft_above',
            'sqft_basement',
            'yr_built',
            'yr_renovated',
            'zipcode',
            'lat',
            'long',
            'sqft_living15',
            'sqft_lot15',
        
        ]
    ]

    pred_log = model.predict(input_data)

    pred_price = np.expm1(pred_log)

    st.metric(
        label="🏡 Estimated House Price",
        value=f"${pred_price[0]:,.0f}"
    )

    st.subheader("Nearest Matching House Used")

    col1, col2 = st.columns(2)

    with col1:
        st.write(f"Bedrooms: {nearest['bedrooms']}")
        st.write(f"Bathrooms: {nearest['bathrooms']}")
        st.write(f"Sqft Living: {nearest['sqft_living']}")

    with col2:
        st.write(f"Grade: {nearest['grade']}")
        st.write(f"View: {nearest['view']}")
        st.write(f"Zipcode: {nearest['zipcode']}")

# -----------------------------------
# Footer
# -----------------------------------

st.info(
    "This model is trained on the King County (USA) housing dataset and it makes predictions based on similar properties."
)
