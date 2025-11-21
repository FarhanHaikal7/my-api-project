from fastapi import FastAPI, HTTPException, Depends, Header
import joblib
import numpy as np
from pydantic import BaseModel, ConfigDict
from typing import List, Annotated
from fastapi.middleware.cors import CORSMiddleware
import firebase_admin
from firebase_admin import credentials, firestore
import os # To find the service account key

# --- 1. INITIALIZE FIREBASE ADMIN ---
# Look for the service account key in the same directory as this script
SERVICE_ACCOUNT_FILE = os.path.join(os.path.dirname(__file__), "serviceAccountKey.json")

db = None # Initialize db to None
try:
    if os.path.exists(SERVICE_ACCOUNT_FILE):
        cred = credentials.Certificate(SERVICE_ACCOUNT_FILE)
        firebase_admin.initialize_app(cred)
        db = firestore.client()
        print("Firebase Admin initialized successfully.")
    else:
        print(f"Error: serviceAccountKey.json not found in {os.path.dirname(__file__)}")
        print("API Key validation will fail.")

except Exception as e:
    print(f"Error initializing Firebase Admin: {e}")
    print("API Key validation will fail.")

# --- 2. Initialize FastAPI app ---
app = FastAPI(title="Breast Cancer Prediction API")

# --- 3. Add CORS Middleware ---
# This allows your index.html website (served from http://127.0.0.1:9000)
# to make requests to this API (running on http://127.0.0.1:8000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows all origins for simplicity, restrict in production
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods (GET, POST, etc.)
    allow_headers=["*"], # Allows all headers (including X-API-Key)
)

# --- 4. Load Model and Scaler ---
# Make sure these files are in the same directory as api.py
try:
    MODEL_FILE = os.path.join(os.path.dirname(__file__), "breast_cancer_model.joblib")
    SCALER_FILE = os.path.join(os.path.dirname(__file__), "scaler.joblib")
    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    print("Model and scaler loaded successfully.")
except FileNotFoundError:
    print(f"Error: Model or scaler files not found in {os.path.dirname(__file__)}")
    print("Please run the script that creates these files first.")
    # You might want to exit here in a real application
    # import sys
    # sys.exit(1)
except Exception as e:
    print(f"Error loading model or scaler: {e}")


TARGET_NAMES = ['malignant', 'benign']

# --- 5. Define Input Data Model ---
class PredictionInput(BaseModel):
    features: List[float]
    model_config = ConfigDict(json_schema_extra={ "example": { "features": [ 17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189 ] } })

# --- 6. API KEY SECURITY DEPENDENCY ---
async def verify_api_key(x_api_key: Annotated[str | None, Header()] = None):
    """
    Checks the 'X-API-Key' header, verifies it against Firestore,
    checks if the user has a 'pro' plan, and returns the user's
    document reference for usage tracking.
    """
    print(f"Received API Key Header: {x_api_key}") # Debugging line

    if db is None:
        print("Error in verify_api_key: Firebase DB not initialized.")
        raise HTTPException(status_code=503, detail="Firebase connection not available. Cannot verify API Key.")

    if x_api_key is None:
        print("Rejecting: X-API-Key header is missing")
        raise HTTPException(status_code=401, detail="X-API-Key header is missing")

    try:
        # Query Firestore for a user profile document containing this API key.
        # This searches across all 'profile' collections under '/artifacts/{appId}/users/{userId}/profile/data'
        # IMPORTANT: This collectionGroup query requires a Firestore index.
        # Firebase might prompt you to create it automatically in the console logs
        # or when you first try to run the query.
        # Index needed: Collection='profile', Field='apiKey', Order=Ascending
        print(f"Searching for key: {x_api_key}")
        users_ref = db.collection_group('profile').where('apiKey', '==', x_api_key).limit(1)
        docs = list(users_ref.stream())

        if not docs:
            print(f"Rejecting: Invalid API Key - {x_api_key}")
            raise HTTPException(status_code=403, detail="Invalid API Key")

        # Key is valid, get user data and document reference
        user_doc = docs[0]
        user_data = user_doc.to_dict()
        user_doc_ref = user_doc.reference
        print(f"API Key belongs to user: {user_doc_ref.parent.parent.id}") # Print the userId

        # Check if the user has the required 'pro' plan
        user_plan = user_data.get('plan')
        if user_plan != 'pro':
            print(f"Rejecting: User plan is '{user_plan}', not 'pro'. User ID: {user_doc_ref.parent.parent.id}")
            raise HTTPException(status_code=403, detail=f"Access denied. Required plan 'pro', found '{user_plan}'. Please contact support.")

        # --- Optional: Add Usage Limit Check ---
        # current_usage = user_data.get('usage', 0)
        # usage_limit = 1000 # Example limit
        # if current_usage >= usage_limit:
        #     print(f"Rejecting: Usage limit exceeded for user {user_doc_ref.parent.parent.id}. Usage: {current_usage}")
        #     raise HTTPException(status_code=429, detail=f"API usage limit ({usage_limit} calls) exceeded.")

        print(f"API Key verified successfully for user: {user_doc_ref.parent.parent.id}, Plan: {user_plan}")
        # If all checks pass, return the document reference
        return user_doc_ref

    except Exception as e:
        print(f"Error during API key verification: {e}")
        # Don't reveal specific internal errors to the client
        raise HTTPException(status_code=500, detail="Internal server error during API key verification.")

# --- 7. PREDICTION ENDPOINT (SECURED) ---
# The `Depends(verify_api_key)` part is CRUCIAL. It forces the security check
# to run BEFORE the predict function can execute.
@app.post("/predict")
async def predict(
    data: PredictionInput,
    user_doc_ref: Annotated[firestore.DocumentReference, Depends(verify_api_key)] # Apply security check
):
    """
    Takes 30 breast cancer features and returns a prediction ('malignant' or 'benign').
    Requires a valid API key with a 'pro' plan via the 'X-API-Key' header.
    """
    print(f"\n--- Received prediction request for user {user_doc_ref.parent.parent.id} ---") # Log request start
    input_features = data.features

    # Validate input length (although Pydantic might handle this partially)
    if len(input_features) != 30:
        print(f"Rejecting: Incorrect number of features. Expected 30, got {len(input_features)}")
        raise HTTPException(status_code=400, detail=f"Expected 30 features, but got {len(input_features)}")

    # Ensure model and scaler are loaded
    if model is None or scaler is None:
         print("Rejecting: Model or scaler not loaded.")
         raise HTTPException(status_code=503, detail="Model service is temporarily unavailable.")

    try:
        # Convert list to 2D numpy array
        input_array = np.array(input_features).reshape(1, -1)
        print("Input features (first 5):", input_array[0, :5])

        # Scale features
        scaled_features = scaler.transform(input_array)
        print("Scaled features (first 5):", scaled_features[0, :5])

        # Make prediction
        prediction_raw = model.predict(scaled_features)
        prediction_index = int(prediction_raw[0])
        prediction_label = TARGET_NAMES[prediction_index]
        print(f"Raw prediction: {prediction_raw}, Index: {prediction_index}, Label: {prediction_label}")

        # Get probabilities
        probabilities = model.predict_proba(scaled_features)
        confidence = float(probabilities[0][prediction_index])
        print(f"Probabilities: {probabilities}, Confidence for predicted class: {confidence:.4f}")

        # --- Secure Usage Tracking ---
        try:
            # Use Firestore atomic increment
            user_doc_ref.update({'usage': firestore.Increment(1)})
            print(f"Usage count incremented for user {user_doc_ref.parent.parent.id}")
        except Exception as e:
            # Log the error but don't stop the prediction from being returned
            print(f"Warning: Failed to update usage count for user {user_doc_ref.parent.parent.id}: {e}")

        # Return the result
        print("--- Prediction request completed successfully ---")
        return {
            "prediction": prediction_label,
            "confidence": f"{confidence:.4f}"
        }

    except ValueError as ve:
        print(f"Prediction error (ValueError): {ve}")
        raise HTTPException(status_code=400, detail=f"Invalid feature value provided: {ve}")
    except Exception as e:
        print(f"Prediction error (Exception): {e}")
        # Avoid sending detailed internal errors to the client
        raise HTTPException(status_code=500, detail="An error occurred during prediction.")

# --- 8. Root Endpoint (Unsecured) ---
@app.get("/")
def read_root():
    """ Provides a simple welcome message. """
    return {"message": "Welcome to the Breast Cancer Prediction API. Use the /predict endpoint with a valid API Key."}

# --- Optional: Run directly for debugging (though uvicorn is preferred) ---
# if __name__ == "__main__":
#     import uvicorn
#     print("Starting server directly with uvicorn (for debugging)...")
#     uvicorn.run(app, host="127.0.0.1", port=8000)



    

