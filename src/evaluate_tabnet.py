# evaluate_tabnet.py

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from pytorch_tabnet.tab_model import TabNetRegressor

def main():
    # 1) Load cleaned transaction data
    df = pd.read_csv("../data/transactions_clean.csv", parse_dates=["InvoiceDate"])
    print(f"✅ Loaded cleaned data: {df.shape}")

    # 2) Feature engineering
    df["hour"] = df["InvoiceDate"].dt.hour
    df["day_of_week"] = df["InvoiceDate"].dt.dayofweek
    df["month"] = df["InvoiceDate"].dt.month

    # 3) Define features and target
    categorical_cols = ["StockCode", "Country"]
    numerical_cols = ["Quantity", "Price", "hour", "day_of_week", "month"]
    features = categorical_cols + numerical_cols
    target = "TotalPrice"

    # 4) Build X, y
    X = df[features]
    y = df[target].values.reshape(-1, 1)

    # 5) Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"✅ Split data → train: {X_train.shape}, test: {X_test.shape}")

    # 6) Encode categorical features
    X_train[categorical_cols] = X_train[categorical_cols].astype(str)
    X_test[categorical_cols] = X_test[categorical_cols].astype(str)

    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    encoder.fit(X_train[categorical_cols])

    X_train[categorical_cols] = encoder.transform(X_train[categorical_cols])
    X_test[categorical_cols] = encoder.transform(X_test[categorical_cols])
    print(f"✅ Applied OrdinalEncoder for: {categorical_cols}")

    # 7) Convert to numpy arrays
    X_train = X_train.values
    X_test = X_test.values

    # 8) Scale numerical features
    num_start = len(categorical_cols)
    scaler = joblib.load("../models/scaler_num.pkl")
    X_test[:, num_start:] = scaler.transform(X_test[:, num_start:])
    print("✅ Scaled numerical features with saved scaler")

    # 9) Load TabNet model
    reg = TabNetRegressor()
    reg.load_model("../models/tabnet_regressor.zip")
    print("✅ TabNet model loaded")

    # 10) Predict and evaluate
    y_pred = reg.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test.flatten(), y_pred.flatten()))
    print(f"\n🎯 Test RMSE: {rmse:.4f}")

if __name__ == "__main__":
    main()
    print("✅ Finished evaluating TabNet model")
    print("========================================")