# Import necessary libraries
from fastapi import FastAPI, HTTPException
import joblib
import numpy as np
from pydantic import BaseModel, ConfigDict
from typing import List
# --- NEW IMPORT ---
from fastapi.middleware.cors import CORSMiddleware  # Import CORS middleware

# 1. Initialize the FastAPI app
app = FastAPI(title="Breast Cancer Prediction API")

# --- ADD THIS ENTIRE BLOCK ---
# 2. Add CORS Middleware
# This is essential to allow your new website (running in a browser)
# to make requests to your API (running on localhost).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (for local development)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)
# --- END OF NEW BLOCK ---

# 3. Load the trained model and scaler
try:
    model = joblib.load("breast_cancer_model.joblib")
    scaler = joblib.load("scaler.joblib")
except FileNotFoundError:
    raise RuntimeError("Model or scaler files not found. Run breastcancer.py first to create them.")

TARGET_NAMES = ['malignant', 'benign']

# 4. Define the input data model (Pydantic V2)
class PredictionInput(BaseModel):
    features: List[float]

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "features": [
                    17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001,
                    0.1471, 0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4,
                    0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193,
                    25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119,
                    0.2654, 0.4601, 0.1189
                ]
            }
        }
    )

# 5. Define a simple root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Breast Cancer Prediction API. Go to /docs to see the endpoints."}

# 6. Define the prediction endpoint (This is what your website will call)
@app.post("/predict")
async def predict(data: PredictionInput):
    """
    Takes 30 breast cancer features and returns a prediction.
    """
    input_features = data.features

    if len(input_features) != 30:
        raise HTTPException(
            status_code=400,
            detail=f"Expected 30 features, but got {len(input_features)}"
        )

    try:
        input_array = np.array(input_features).reshape(1, -1)
        scaled_features = scaler.transform(input_array)
        prediction_raw = model.predict(scaled_features)
        prediction_index = int(prediction_raw[0])
        prediction_label = TARGET_NAMES[prediction_index]
        
        probabilities = model.predict_proba(scaled_features)
        confidence = float(probabilities[0][prediction_index])
    
        return {
            "prediction": prediction_label,
            "confidence": f"{confidence:.4f}"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")
