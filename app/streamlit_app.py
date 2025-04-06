# app/streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.metrics import mean_squared_error

# ─── 1. Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="🛒 E‑commerce Pricing Predictor",
    page_icon="💰",
    layout="wide",
)

st.markdown("""
    <style>
      /* Adjust sidebar width */
      .sidebar .sidebar-content { width: 200px; }
      /* Hide the “Made with Streamlit” footer */
      footer { visibility: hidden; }
    </style>
    """, unsafe_allow_html=True)

# ─── 2. Load Artifacts ────────────────────────────────────────────
@st.cache_resource
def load_model():
    m = TabNetRegressor()
    m.load_model("src/models/tabnet_regressor.zip.zip")
    return m

@st.cache_resource
def load_encoders():
    return {
        "StockCode": joblib.load("models/enc_StockCode.pkl"),
        "Country":   joblib.load("models/enc_Country.pkl")
    }

@st.cache_resource
def load_scaler():
    return joblib.load("src/models/scaler_num.pkl")

@st.cache_data
def load_data():
    df = pd.read_csv("data/transactions_clean.csv", parse_dates=["InvoiceDate"])
    return df

model    = load_model()
encoders = load_encoders()
scaler   = load_scaler()
df_full  = load_data()

# ─── 3. Prepare Dropdown Options ─────────────────────────────────
stock_codes = sorted(df_full["StockCode"].unique().tolist())
countries   = sorted(df_full["Country"].unique().tolist())

# ─── 4. Sidebar Navigation ────────────────────────────────────────
st.sidebar.title("Navigation")
page = st.sidebar.radio("", ["Home", "Single Prediction", "Batch Prediction"])

# ─── 5. Helper: Preprocess Single Record ─────────────────────────
def preprocess(record: dict) -> np.ndarray:
    # categorical
    cat_vals = []
    for col in ["StockCode", "Country"]:
        enc = encoders[col]
        try:
            cat_vals.append(int(enc.transform([[record[col]]])[0][0]))
        except:
            cat_vals.append(-1)
    # numeric
    num_vals = [
        record["Quantity"], record["Price"],
        record["hour"], record["day_of_week"], record["month"]
    ]
    arr = np.array(cat_vals + num_vals, dtype=float).reshape(1, -1)
    arr[:, len(cat_vals):] = scaler.transform(arr[:, len(cat_vals):])
    return arr

# ─── 6. Home Page ─────────────────────────────────────────────────
if page == "Home":
    st.title("🛒 E‑commerce Pricing Predictor")
    st.markdown(
        """
        This dashboard lets you predict transaction TotalPrice using a pretrained TabNet model.
        Navigate via the sidebar to run single or batch predictions.
        """
    )
    # Show RMSE on hold‑out test
    # Compute once and cache
    @st.cache_data
    def compute_rmse():
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error
        from sklearn.preprocessing import OrdinalEncoder
        import numpy as np
        import pandas as pd

        # 1) Reload & recreate all features from scratch
        df = pd.read_csv("data/transactions_clean.csv", parse_dates=["InvoiceDate"])
        # Time features
        df["hour"]        = df["InvoiceDate"].dt.hour
        df["day_of_week"] = df["InvoiceDate"].dt.dayofweek
        df["month"]       = df["InvoiceDate"].dt.month

        # 2) Define feature columns (make sure these match your CSV!)
        feats = ["StockCode", "Country", "Quantity", "Price", "hour", "day_of_week", "month"]
        X = df[feats]
        y = df["TotalPrice"].values

        # 3) Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # 4) Encode categoricals robustly
        cat_cols = ["StockCode", "Country"]
        oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        oe.fit(X_train[cat_cols].astype(str))
        X_test[cat_cols] = oe.transform(X_test[cat_cols].astype(str))

        # 5) Scale numeric slice
        X_test_np = X_test.to_numpy()
        num_start = len(cat_cols)
        X_test_np[:, num_start:] = scaler.transform(X_test_np[:, num_start:])

        # 6) Predict & score
        preds = model.predict(X_test_np).flatten()
        rmse  = np.sqrt(mean_squared_error(y_test, preds))
        return rmse
    rmse = compute_rmse()
    st.metric("Test RMSE", f"{rmse:.2f}")
    # Calculate KPIs from the full dataset
    total_sales = df_full["TotalPrice"].sum()
    avg_order   = df_full["TotalPrice"].mean()

    # Display the KPIs
    st.metric("💰 Total Sales", f"${total_sales:,.0f}")
    st.metric("📈 Avg. Order Value", f"${avg_order:.2f}")

    # Data overview
    st.subheader("Sample Transactions")
    st.dataframe(df_full.sample(10), use_container_width=True)
    
    # Distribution of TotalPrice
    st.subheader("Distribution of Transaction TotalPrice")

    # Quick summary stats for debugging
    st.write("**TotalPrice Summary Stats**")
    st.write(df_full["TotalPrice"].describe())

    # Drop NaNs or zero if needed
    df_cleaned = df_full[df_full["TotalPrice"] > 0].copy()

    # Calculate 99th percentile for trimming
    q99 = df_cleaned["TotalPrice"].quantile(0.99)

    # Trim at 99th percentile (to remove extreme outliers)
    df_trimmed = df_cleaned[df_cleaned["TotalPrice"] <= q99]

    # Fallback check: Make sure trimming didn't remove too much
    if df_trimmed.empty:
        st.warning("Trimmed data is empty. Falling back to log scale without trimming.")
        fig = px.histogram(df_cleaned, x="TotalPrice", nbins=50, log_x=True,
                        color_discrete_sequence=["#636EFA"])
        fig.update_layout(title="TotalPrice Distribution (Log Scale)", margin=dict(l=0,r=0,t=30,b=0))
        st.plotly_chart(fig, use_container_width=True)

    else:
        # Plot trimmed data (normal scale)
        fig = px.histogram(df_trimmed, x="TotalPrice", nbins=50,
                        color_discrete_sequence=["#636EFA"])
        fig.update_layout(title="TotalPrice Distribution (Trimmed at 99th Percentile)", margin=dict(l=0,r=0,t=30,b=0))
        st.plotly_chart(fig, use_container_width=True)

        # Optional: Also show log scale version
        fig_log = px.histogram(df_cleaned, x="TotalPrice", nbins=50, log_x=True,
                            color_discrete_sequence=["#EF553B"])
        

    
    # Example: Create a monthly sales trend chart
    monthly_sales = df_full.copy()
    monthly_sales["Month"] = monthly_sales["InvoiceDate"].dt.to_period("M").astype(str)
    monthly_sales = monthly_sales.groupby("Month")["TotalPrice"].sum().reset_index()

    fig = px.line(monthly_sales, x="Month", y="TotalPrice",
                title="Monthly Sales Trend",
                color_discrete_sequence=["#4CAF50"])
    fig.update_layout(xaxis_title=None, yaxis_title="Sales ($)", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)


# ─── 7. Single Prediction ─────────────────────────────────────────
elif page == "Single Prediction":
    st.title("💡 Single Transaction Prediction")
    with st.form("single_form", clear_on_submit=False):
        col1, col2, col3 = st.columns(3)
        st.markdown("#### Transaction Details")
        st.markdown("Fill in the details below to get a prediction.")
        with col1:
            stock        = st.selectbox("StockCode", stock_codes, label_visibility="collapsed")
            country      = st.selectbox("Country", countries, label_visibility="collapsed")
            quantity     = st.number_input("Quantity", min_value=1, value=1)
        with col2:
            unit_price   = st.number_input("Price", min_value=0.0, value=1.0, format="%.2f")
            hour         = st.slider("Hour of Day", 0, 23, 12)
            day_of_week  = st.selectbox("Day of Week", list(range(7)))
        with col3:
            month        = st.selectbox("Month", list(range(1,13)))
            submitted    = st.form_submit_button("Predict")
    if submitted:
        record = {
            "StockCode": stock,
            "Country": country,
            "Quantity": quantity,
            "Price": unit_price,
            "hour": hour,
            "day_of_week": day_of_week,
            "month": month
        }
        X = preprocess(record)
        pred = model.predict(X)[0][0]
        st.success(f"Predicted Total Price: **{pred:.2f}**")
        # Bonus: show gauge
        st.metric("💰 Predicted TotalPrice", f"${pred:.2f}")

# ─── 8. Batch Prediction ──────────────────────────────────────────
elif page == "Batch Prediction":
    st.title("📁 Batch Prediction")
    st.markdown("Upload a CSV with columns: StockCode, Country, Quantity, Price, hour, day_of_week, month")
    uploaded = st.file_uploader("Upload CSV", type="csv")
    if uploaded:
        df_batch = pd.read_csv(uploaded)
        st.subheader("Uploaded Data Sample")
        st.dataframe(df_batch.head(), use_container_width=True)
        req_cols = ["StockCode","Country","Quantity","Price","hour","day_of_week","month"]
        if not all(c in df_batch.columns for c in req_cols):
            st.error(f"Missing columns! Required: {req_cols}")
        else:
            # preprocess
            X_list = [preprocess(row) for _, row in df_batch[req_cols].iterrows()]
            X_batch = np.vstack(X_list)
            preds   = model.predict(X_batch).flatten()
            df_batch["PredictedTotalPrice"] = preds
            st.success("✅ Predictions added")
            st.dataframe(df_batch, use_container_width=True)
            # If actual exists, show eval plots
            if "TotalPrice" in df_batch.columns:
                df_batch["Error"] = df_batch["PredictedTotalPrice"] - df_batch["TotalPrice"]
                # Actual vs Predicted
                fig1 = px.scatter(df_batch, x="TotalPrice", y="PredictedTotalPrice",
                                  trendline="ols", title="Actual vs Predicted")
                st.plotly_chart(fig1, use_container_width=True)
                # Error distribution
                fig2 = px.histogram(df_batch, x="Error", nbins=50,
                                    title="Prediction Error Distribution",
                                    color_discrete_sequence=["#EF553B"])
                st.plotly_chart(fig2, use_container_width=True)
