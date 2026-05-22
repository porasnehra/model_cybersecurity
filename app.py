from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import uvicorn
from typing import Dict, Any

app = FastAPI(title="Mule Account Detection API", description="API to predict if a transaction/account is a mule account.")

model = None
imputer = None
numeric_features = None

@app.on_event("startup")
def load_model():
    global model, imputer, numeric_features
    try:
        model = joblib.load('mule_account_model.pkl')
        imputer = joblib.load('imputer.pkl')
        numeric_features = joblib.load('numeric_features.pkl')
        print("Models loaded successfully.")
    except Exception as e:
        print(f"Error loading models: {e}. Make sure to train the model first.")

class TransactionData(BaseModel):
    # Expecting a dictionary of feature names to values
    features: Dict[str, float]

@app.post("/predict")
def predict(data: TransactionData):
    if model is None or imputer is None or numeric_features is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")
    
    try:
        row = []
        for feat in numeric_features:
            row.append(data.features.get(feat, 0.0))
            
        df_input = pd.DataFrame([row], columns=numeric_features)
        X_input = imputer.transform(df_input)
        
        prediction = model.predict(X_input)
        probability = model.predict_proba(X_input)[0][1]
        
        return {
            "is_mule_account": bool(prediction[0] == 1),
            "risk_score": float(probability),
            "message": "Prediction successful"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
