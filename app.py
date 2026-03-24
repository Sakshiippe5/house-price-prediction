import streamlit as st
import joblib
import numpy as np

# Load model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

# Page config
st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="wide")

# 🎨 Custom CSS
st.markdown("""
<style>

/* Full page background */
.stApp {
    background: linear-gradient(135deg, #e0ecff, #f7f9fc);
}

/* Center card */
.main-container {
    max-width: 800px;
    margin: auto;
    padding: 2rem;
    background: white;
    border-radius: 18px;
    box-shadow: 0px 10px 30px rgba(0,0,0,0.1);
}

/* Title */
h1 {
    text-align: center;
    color: #2c3e50;
}

/* Sliders */
.stSlider > div > div > div {
    background-color: #4CAF50 !important; /* GREEN BAR */
}

.stSlider > div > div > div > div {
    background-color: #4CAF50 !important; /* knob */
}

/* Button */
.stButton>button {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 18px;
    border: none;
}

.stButton>button:hover {
    opacity: 0.9;
}

/* Metric */
[data-testid="stMetricValue"] {
    font-size: 36px;
    color: #27ae60;
    text-align: center;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

# Wrap everything in centered container
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# Title
st.markdown("<h1>🏠 House Price Predictor</h1>", unsafe_allow_html=True)
st.write("Enter the details below to predict the house price.")

# Inputs (2-column layout INSIDE center card)
col1, col2 = st.columns(2)

with col1:
    med_inc = st.slider("Median Income", 0.5, 15.0, 3.0)
    house_age = st.slider("House Age", 1, 52, 20)
    ave_rooms = st.slider("Average Rooms", 1.0, 15.0, 5.0)
    ave_bedrms = st.slider("Average Bedrooms", 1.0, 5.0, 1.0)

with col2:
    population = st.slider("Population", 3, 3500, 1000)
    ave_occup = st.slider("Occupants per House", 1.0, 6.0, 2.5)
    latitude = st.slider("Latitude", 32.5, 42.0, 36.0)
    longitude = st.slider("Longitude", -124.0, -114.0, -119.0)

# Feature Engineering
rooms_per_person = ave_rooms / ave_occup
bedrooms_ratio = ave_bedrms / ave_rooms
income_rooms = med_inc * ave_rooms

features = np.array([[med_inc, house_age, ave_rooms, ave_bedrms,
                      population, ave_occup, latitude, longitude,
                      rooms_per_person, bedrooms_ratio, income_rooms]])

# Predict button
if st.button("💰 Predict Price"):
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    price = prediction * 100000

    st.markdown("---")

    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.metric("Estimated House Price", f"${price:,.0f}")

    st.success("Prediction Complete ✅")

# Footer
st.markdown("---")
st.caption("Model: Random Forest | R² = 0.7768")

# Close container
st.markdown('</div>', unsafe_allow_html=True)