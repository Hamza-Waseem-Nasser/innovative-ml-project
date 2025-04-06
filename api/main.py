# api/main.py
from fastapi import FastAPI, HTTPException
from api.schemas import TransactionFeatures, PredictionResponse
from api.utils import preprocess_input, model

app = FastAPI(title="TabNet E‑commerce Pricing API")

@app.get("/")
def health_check():
    return {"status": "OK", "message": "API is running"}

@app.post("/predict", response_model=PredictionResponse)
def predict_price(payload: TransactionFeatures):
    try:
        # Convert Pydantic model to dict
        data = payload.dict()
        # Preprocess
        X = preprocess_input(data)
        # Predict
        pred = model.predict(X)[0][0]  # get scalar
        return {"TotalPrice": float(pred)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
