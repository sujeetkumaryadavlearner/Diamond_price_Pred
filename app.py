import streamlit as st
import numpy as np
import pickle

# Load the saved model and scalers
model = pickle.load(open("model.pkl", "rb"))
x_scaler = pickle.load(open("scaler_x.pkl", "rb"))

# Streamlit UI
st.title("Diamond Price Prediction")

with st.form("diamond_form"):
    carat = st.sidebar.number_input("Carat", value=None, placeholder="Enter carat value")
    cut = st.sidebar.selectbox("Cut", ["Fair", "Good", "Very Good", "Premium", "Ideal"], index=None, placeholder="Select cut")
    color = st.sidebar.selectbox("Color", ["D", "E", "F", "G", "H", "I", "J"], index=None, placeholder="Select color")
    clarity = st.sidebar.selectbox("Clarity", ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"], index=None, placeholder="Select clarity")
    depth = st.sidebar.number_input("Depth %", value=None, placeholder="Enter depth %")
    table = st.sidebar.number_input("Table %", value=None, placeholder="Enter table %")
    x = st.sidebar.number_input("X (length in mm)", value=None, placeholder="Enter length")
    y = st.sidebar.number_input("Y (width in mm)", value=None, placeholder="Enter width")
    z = st.sidebar.number_input("Z (depth in mm)", value=None, placeholder="Enter depth")

    # Submit button
    submitted = st.form_submit_button("Predict Price")

# Encode categorical features

if submitted:
    if None in [carat, depth, table, x, y, z] or cut == "" or color == "" or clarity == "":
        st.error("Please fill all the fields before submitting.")
    else:
        st.success(f"Predicting price for: Carat={carat}, Cut={cut}, Color={color}, Clarity={clarity}, Depth={depth}, Table={table}, X={x}, Y={y}, Z={z}")
        cut_map = {"Fair": 1, "Good": 2, "Very Good": 3, "Premium": 4, "Ideal": 5}
        color_map = {"D": 1, "E": 2, "F": 3, "G": 4, "H": 5, "I": 6, "J": 7}
        clarity_map = {"I1": 1, "SI2": 2, "SI1": 3, "VS2": 4, "VS1": 5, "VVS2": 6, "VVS1": 7, "IF": 8}
        cut_encoded = cut_map[cut]
        color_encoded = color_map[color]
        clarity_encoded = clarity_map[clarity]
        features = np.array([[carat, cut_encoded, color_encoded, clarity_encoded, depth, table, x, y, z]])
        features_scaled = x_scaler.transform(features)
        predicted_price = model.predict(features_scaled)
        st.subheader("Predicted Diamond Price 💰")
        st.write(f"**₹ {predicted_price[0][0]:,.2f}**")
        st.success(f"Predicting price for: Carat={carat}, Cut={cut}, Color={color}, Clarity={clarity}, Depth={depth}, Table={table}, X={x}, Y={y}, Z={z}")
        # Here, you can call your prediction model
# Scale the input data

