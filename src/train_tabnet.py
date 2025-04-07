import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from pytorch_tabnet.pretraining import TabNetPretrainer
from pytorch_tabnet.tab_model import TabNetRegressor
import torch
import joblib

# 1) Load the cleaned transactions
df = pd.read_csv("../data/transactions_clean.csv", parse_dates=["InvoiceDate"])
print("Loaded transactions:", df.shape)

# 2) Feature Engineering
# ──────────────────────
# Extract time features
df["hour"] = df["InvoiceDate"].dt.hour
df["day_of_week"] = df["InvoiceDate"].dt.dayofweek
df["month"] = df["InvoiceDate"].dt.month

df.info()

# Define features and target
target = "TotalPrice"
categorical_cols = ["StockCode", "Country"]
numerical_cols   = ["Quantity", "Price", "hour", "day_of_week", "month"]
features = categorical_cols + numerical_cols

# 3) Encode categoricals
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    joblib.dump(le, f"../models/enc_{col}.pkl")  # save encoders


    # 4) Prepare data arrays
X = df[features].values
y = df[target].values.reshape(-1, 1)

# 5) Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("Train size:", X_train.shape, "Test size:", X_test.shape)

# 6) Scale numerical features
#    We scale only the numerical part of X
num_start = len(categorical_cols)
scaler = StandardScaler()
X_train[:, num_start:] = scaler.fit_transform(X_train[:, num_start:])
X_test[:, num_start:]  = scaler.transform(X_test[:, num_start:])
joblib.dump(scaler, "../models/scaler_num.pkl")

# 7) Self‑Supervised Pretraining
# ────────────────────────────────
from pytorch_tabnet.callbacks import EarlyStopping

pretrainer = TabNetPretrainer(
    input_dim=X_train.shape[1],
    mask_type='entmax'  # sparsity in masks
)
# define callbacks
es = EarlyStopping(
    patience=20,
    early_stopping_metric="loss",  # Metric to monitor
    is_maximize=False              # Set to False for minimizing loss
)

# fit with callbacks and verbose logging
pretrainer.fit(
    X_train,
    max_epochs=100,
    batch_size=1024,
    virtual_batch_size=128,
    pretraining_ratio=0.8,
    num_workers=3,
    drop_last=False,
    callbacks=[es]
    
)

pretrainer.save_model("../models/tabnet_pretrainer")
print("✅ Pretraining complete.")

# 8) Fine‑Tuning as Regressor
# ────────────────────────────
regressor = TabNetRegressor(
    input_dim=X_train.shape[1],
    output_dim=1,
    optimizer_fn=torch.optim.Adam,
    optimizer_params={"lr":1e-3, "weight_decay":1e-4},
    scheduler_params={"step_size":50, "gamma":0.9},
    scheduler_fn=torch.optim.lr_scheduler.StepLR
)

# load pretrained encoder weights
# Instead of regressor.load_weights_from_unsupervised(pretrainer)
# use the following:
regressor._set_network() # Initialize the network attribute
regressor.load_weights_from_unsupervised(pretrainer) # Load weights


# fit on train, evaluate on test
regressor.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    eval_name=['test'],
    max_epochs=100,
    patience=30,
    batch_size=1024,
    virtual_batch_size=128,
    num_workers=3,
    drop_last=False
)

# Save the fine‑tuned model
regressor.save_model("../models/tabnet_regressor")
print("✅ Fine‑tuning complete. Model saved to models/tabnet_regressor.zip")