# api/utils.py
import joblib
import numpy as np
from pytorch_tabnet.tab_model import TabNetRegressor

# Paths to your saved artifacts
ENCODERS = {
    "StockCode": "models/enc_StockCode.pkl",
    "Country":   "models/enc_Country.pkl"
}
SCALER_PATH = "models/scaler_num.pkl"
MODEL_PATH  = "models/tabnet_regressor.zip"

# Load encoders
encoders = {col: joblib.load(path) for col, path in ENCODERS.items()}

# Load scaler
scaler = joblib.load(SCALER_PATH)

# Load TabNet model
model = TabNetRegressor()
model.load_model(MODEL_PATH)

def preprocess_input(data: dict) -> np.ndarray:
    """
    Takes a dict of features, applies encoding and scaling,
    and returns a 2D numpy array ready for model.predict().
    """
    # Order must match training features
    cat_cols = ["StockCode", "Country"]
    num_cols = ["Quantity", "Price", "hour", "day_of_week", "month"]
    
    # Encode categoricals
    cat_vals = []
    for col in cat_cols:
        val = data[col]
        # transform unknowns to -1
        enc = encoders[col]
        try:
            cat_vals.append(int(enc.transform([[val]])[0][0]))
        except:
            cat_vals.append(-1)
    
    # Numerical values
    num_vals = [data[col] for col in num_cols]
    
    # Combine and scale numeric slice
    arr = np.array(cat_vals + num_vals, dtype=float).reshape(1, -1)
    # scale numeric part in-place
    arr[:, len(cat_cols):] = scaler.transform(arr[:, len(cat_cols):])
    return arr
