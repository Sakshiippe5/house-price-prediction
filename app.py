import streamlit as st
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

@st.cache_resource
def train_model():
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame
    df = df[df['MedHouseVal'] < 5.0]
    df = df[df['AveRooms'] < 50]
    df = df[df['AveOccup'] < 10]
    df['rooms_per_person'] = df['AveRooms'] / df['AveOccup']
    df['bedrooms_ratio'] = df['AveBedrms'] / df['AveRooms']
    df['income_rooms'] = df['MedInc'] * df['AveRooms']
    X = df.drop('MedHouseVal', axis=1)
    y = df['MedHouseVal']
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train_scaled, y_train)
    return model, scaler

# Page config
st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="wide")

# Your custom CSS
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #e0ecff, #f7f9fc);
}
.main-container {
    max-width: 800px;
    margin: auto;
    padding: 2rem;
    background: white;
    border-radius: 18px;
    box-shadow: 0px 10px 30px rgba(0,0,0,0.1);
}
h1 {
    text-align: center;
    color: #2c3e50;
}
.stSlider > div > div > div {
    background-color: #4CAF50 !important;
}
.stSlider > div > div > div > div {
    background-color: #4CAF50 !important;
}
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
[data-testid="stMetricValue"] {
    font-size: 36px;
    color: #27ae60;
    text-align: center;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-container">', unsafe_allow_html=True)
st.markdown("<h1>🏠 House Price Predictor</h1>", unsafe_allow_html=True)
st.write("Enter the details below to predict the house price.")

# Load model
with st.spinner("Loading model... (first load takes ~30 seconds)"):
    model, scaler = train_model()

# Inputs
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

# Feature engineering
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
    col_a, col_b, col_c = st.columns([1,2,1])
    with col_b:
        st.metric("Estimated House Price", f"${price:,.0f}")
    st.success("Prediction Complete ✅")

st.markdown("---")
st.caption("Model: Random Forest | R² = 0.7768 | California Housing Dataset")
st.markdown('</div>', unsafe_allow_html=True)


