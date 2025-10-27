import streamlit as st
import tensorflow as tf
import joblib
import numpy as np

# ============================================
# Load models and preprocessors
# ============================================
@st.cache_resource
def load_models():
    traffic_model = tf.keras.models.load_model("models/traffic_density_model.h5")
    accident_model = tf.keras.models.load_model("models/accident_prediction_model.h5")
    signal_model = tf.keras.models.load_model("models/signal_optimization_model.h5")
    scaler = joblib.load("models/scaler.pkl")
    label_encoder = joblib.load("models/label_encoder.pkl")
    return traffic_model, accident_model, signal_model, scaler, label_encoder

traffic_model, accident_model, signal_model, scaler, label_encoder = load_models()

# ============================================
# Streamlit UI
# ============================================
st.title("🚦 AI Smart Indian Traffic System")

hour = st.slider("Select Hour (0–23)", 0, 23, 12)
day = st.selectbox("Day", label_encoder.classes_)
weather = st.selectbox("Weather", ['Sunny', 'Rainy', 'Cloudy'])
vehicle_count = st.number_input("Vehicle Count", 0, 1000, 200)
accidents = st.number_input("Past Accidents at Location", 0, 10, 1)

# Preprocess inputs
day_encoded = label_encoder.transform([day])[0]
weather_encoded = label_encoder.transform([weather])[0]

# For predictions
input_density = np.array([[hour, day_encoded, weather_encoded]])
input_accident = np.array([[hour, day_encoded, weather_encoded, vehicle_count]])
input_signal = np.array([[hour, day_encoded, weather_encoded, vehicle_count, accidents]])

scaled_density = scaler.transform(np.pad(input_density, ((0,0),(0,3)), 'constant'))
scaled_accident = scaler.transform(np.pad(input_accident, ((0,0),(0,2)), 'constant'))
scaled_signal = scaler.transform(np.pad(input_signal, ((0,0),(0,1)), 'constant'))

# Predictions
pred_density = traffic_model.predict(scaled_density)[0][0]
pred_accident = accident_model.predict(scaled_accident)[0][0]
pred_signal = signal_model.predict(scaled_signal)[0][0]

st.subheader("🔮 Predictions:")
st.write(f"🚗 **Traffic Density:** {pred_density:.2f}")
st.write(f"⚠️ **Accident Probability:** {pred_accident*100:.2f}%")
st.write(f"🕒 **Optimized Signal Time:** {pred_signal*100:.2f} sec")
