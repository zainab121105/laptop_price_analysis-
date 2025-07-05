import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

st.title("💻 Laptop Price Predictor (INR)")

# Load and preprocess data
@st.cache_data
def load_and_train():
    df = pd.read_csv('laptop_prices.csv')
    binary_cols = ['Touchscreen', 'IPSpanel', 'RetinaDisplay']
    for col in binary_cols:
        df[col] = df[col].map({'Yes': 1, 'No': 0})
    df['ScreenArea'] = df['ScreenW'] * df['ScreenH']
    df['TotalStorage'] = df['PrimaryStorage'] + df['SecondaryStorage']
    df = df.drop(columns=['Product', 'GPU_model', 'CPU_model', 'Screen', 'ScreenW', 'ScreenH'])
    categorical_cols = ['Company', 'TypeName', 'OS', 'PrimaryStorageType', 
                        'SecondaryStorageType', 'CPU_company', 'GPU_company']
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    df['Price_inr'] = df['Price_euros'] * 90
    X = df.drop(['Price_euros', 'Price_inr'], axis=1)
    y = df['Price_inr']
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, df

model, df = load_and_train()

# User input form
st.header("Enter Laptop Specs:")
col1, col2 = st.columns(2)
with col1:
    inches = st.number_input("Screen Size (Inches)", 10.0, 20.0, 15.6)
    ram = st.number_input("RAM (GB)", 2, 64, 8)
    weight = st.number_input("Weight (Kg)", 0.5, 5.0, 1.5)
    cpu_freq = st.number_input("CPU Frequency (GHz)", 1.0, 5.0, 2.5)
with col2:
    touchscreen = st.selectbox("Touchscreen", [0, 1])
    ips = st.selectbox("IPS Panel", [0, 1])
    retina = st.selectbox("Retina Display", [0, 1])
    screen_area = st.number_input("Screen Area (width x height)", 100000, 5000000, 2000000)
    total_storage = st.number_input("Total Storage (GB)", 128, 4000, 512)

if st.button("Predict Price"):
    sample = np.array([inches, ram, weight, cpu_freq, touchscreen, ips, retina, screen_area, total_storage])
    extra_zeros = np.zeros(model.n_features_in_ - len(sample))
    sample_full = np.concatenate([sample, extra_zeros]).reshape(1, -1)
    price_pred = model.predict(sample_full)[0]
    st.success(f"💰 Predicted Price: ₹{int(price_pred):,}")