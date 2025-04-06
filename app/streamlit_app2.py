# app/streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests

# ─── 1. Page Config & Custom CSS ───────────────────────────────────────────────
st.set_page_config(
    page_title="✨ E‑commerce Pricing Predictor",
    page_icon="💸",
    layout="wide",
)

# Custom CSS for new color scheme and design
st.markdown("""
    <style>
        /* Overall background and font settings */
        body {
            background-color: #f0f2f6;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        /* Sidebar customizations */
        .css-1d391kg {  /* container */
            background-color: #2C3E50 !important;
            color: #ecf0f1;
        }
        .css-1d391kg .css-1d391kg {  
            background-color: #2C3E50;
        }
        .css-1d391kg .css-1d391kg h1, .css-1d391kg .css-1d391kg p {
            color: #ecf0f1;
        }
        /* Sidebar title and radio button */
        .sidebar .sidebar-content { 
            background-color: #2C3E50; 
        }
        .css-1aumxhk { 
            color: #ecf0f1; 
        }
        /* Hide Streamlit footer */
        footer { visibility: hidden; }
        /* Custom header style */
        .main h1 {
            color: #34495e;
            font-weight: 600;
        }
        /* Metric style */
        .stMetric { background-color: #3498db; color: #fff; border-radius: 10px; padding: 10px; }
        /* Buttons */
        .stButton>button {
            background-color: #e74c3c;
            color: white;
            border: none;
            padding: 0.5em 1em;
            border-radius: 5px;
        }
    </style>
    """, unsafe_allow_html=True)

# ─── 2. Load Data ────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("data/transactions_clean.csv", parse_dates=["InvoiceDate"])
    return df

df_full = load_data()

# ─── 3. Prepare Dropdown Options ─────────────────────────────────
stock_codes = sorted(df_full["StockCode"].unique().tolist())
countries   = sorted(df_full["Country"].unique().tolist())

# ─── 4. Sidebar Navigation ───────────────────────────────────────
st.sidebar.title("🗺 Navigation")
page = st.sidebar.radio("Select a Page", ["Home", "Single Prediction", "Batch Prediction"])

# ─── 5. Helper: Call the Prediction API ─────────────────────────
def call_prediction_api(record: dict) -> float:
    """
    Sends the record as JSON to the prediction API endpoint and returns the predicted TotalPrice.
    """
    url = "http://127.0.0.1:8000/predict"  # Ensure your API is running here
    response = requests.post(url, json=record)
    response.raise_for_status()  # Raise error if request failed
    result = response.json()
    return result["TotalPrice"]

# ─── 6. Home Page ─────────────────────────────────────────────────
if page == "Home":
    st.title("✨ E‑commerce Pricing Predictor")
    st.markdown("""
        Welcome to the sleek dashboard for predicting transaction TotalPrice via our TabNet model.
        Use the sidebar to navigate to single or batch prediction pages.
    """)
    
    # Display KPIs from local data
    total_sales = df_full["TotalPrice"].sum()
    avg_order   = df_full["TotalPrice"].mean()
    col1, col2 = st.columns(2)
    col1.metric("💰 Total Sales", f"${total_sales:,.0f}")
    col2.metric("📈 Avg. Order Value", f"${avg_order:.2f}")
    
    # Data overview
    st.subheader("Sample Transactions")
    st.dataframe(df_full.sample(10), use_container_width=True)
    
    # Distribution of TotalPrice
    st.subheader("TotalPrice Distribution")
    st.write("**Summary Stats**")
    st.write(df_full["TotalPrice"].describe())
    
    # Remove extreme outliers and display histogram
    df_cleaned = df_full[df_full["TotalPrice"] > 0].copy()
    q99 = df_cleaned["TotalPrice"].quantile(0.99)
    df_trimmed = df_cleaned[df_cleaned["TotalPrice"] <= q99]
    if df_trimmed.empty:
        st.warning("Trimmed data is empty. Displaying log scale.")
        fig = px.histogram(df_cleaned, x="TotalPrice", nbins=50, log_x=True, color_discrete_sequence=["#2980b9"])
        fig.update_layout(title="TotalPrice Distribution (Log Scale)", margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig = px.histogram(df_trimmed, x="TotalPrice", nbins=50, color_discrete_sequence=["#2980b9"])
        fig.update_layout(title="TotalPrice Distribution (Trimmed)", margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)
        
    
    # Monthly sales trend
    monthly_sales = df_full.copy()
    monthly_sales["Month"] = monthly_sales["InvoiceDate"].dt.to_period("M").astype(str)
    monthly_sales = monthly_sales.groupby("Month")["TotalPrice"].sum().reset_index()
    fig_line = px.line(monthly_sales, x="Month", y="TotalPrice",
                       title="Monthly Sales Trend", color_discrete_sequence=["#8e44ad"])
    fig_line.update_layout(xaxis_title=None, yaxis_title="Sales ($)", template="plotly_white")
    st.plotly_chart(fig_line, use_container_width=True)

# ─── 7. Single Prediction ─────────────────────────────────────────
elif page == "Single Prediction":
    st.title("🔍 Single Transaction Prediction")
    with st.form("single_form", clear_on_submit=False):
        col1, col2, col3 = st.columns(3)
        st.markdown("#### Transaction Details")
        st.markdown("Enter the details below to get a prediction.")
        with col1:
            stock    = st.selectbox("StockCode", stock_codes, label_visibility="collapsed")
            country  = st.selectbox("Country", countries, label_visibility="collapsed")
            quantity = st.number_input("Quantity", min_value=1, value=1)
        with col2:
            price      = st.number_input("Price", min_value=0.0, value=1.0, format="%.2f")
            hour       = st.slider("Hour (0-23)", 0, 23, 12)
            day_of_week = st.selectbox("Day of Week", list(range(7)))
        with col3:
            month    = st.selectbox("Month", list(range(1, 13)))
            submitted = st.form_submit_button("Predict", help="Click to get the prediction")
    if submitted:
        record = {
            "StockCode": stock,
            "Country": country,
            "Quantity": quantity,
            "Price": price,
            "hour": hour,
            "day_of_week": day_of_week,
            "month": month
        }
        try:
            pred = call_prediction_api(record)
            st.success(f"Predicted Total Price: **${pred:.2f}**")
            st.metric("💸 Predicted Price", f"${pred:.2f}")
        except Exception as e:
            st.error(f"Error calling prediction API: {e}")

# ─── 8. Batch Prediction ──────────────────────────────────────────
elif page == "Batch Prediction":
    st.title("📂 Batch Prediction")
    st.markdown("Upload a CSV with columns: StockCode, Country, Quantity, Price, hour, day_of_week, month")
    uploaded = st.file_uploader("Upload CSV", type="csv")
    if uploaded:
        df_batch = pd.read_csv(uploaded)
        st.subheader("Uploaded Data Sample")
        st.dataframe(df_batch.head(), use_container_width=True)
        req_cols = ["StockCode", "Country", "Quantity", "Price", "hour", "day_of_week", "month"]
        if not all(col in df_batch.columns for col in req_cols):
            st.error(f"Missing columns! Required: {req_cols}")
        else:
            predictions = []
            for _, row in df_batch[req_cols].iterrows():
                record = row.to_dict()
                try:
                    pred = call_prediction_api(record)
                    predictions.append(pred)
                except Exception as e:
                    predictions.append(np.nan)
            df_batch["PredictedTotalPrice"] = predictions
            st.success("✅ Predictions added")
            st.dataframe(df_batch, use_container_width=True)
            if "TotalPrice" in df_batch.columns:
                df_batch["Error"] = df_batch["PredictedTotalPrice"] - df_batch["TotalPrice"]
                fig1 = px.scatter(df_batch, x="TotalPrice", y="PredictedTotalPrice",
                                  trendline="ols", title="Actual vs Predicted")
                st.plotly_chart(fig1, use_container_width=True)
                fig2 = px.histogram(df_batch, x="Error", nbins=50,
                                    title="Prediction Error Distribution",
                                    color_discrete_sequence=["#c0392b"])
                st.plotly_chart(fig2, use_container_width=True)
