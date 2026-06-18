import os
import pickle
import numpy as np
import urllib.request

from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Initialize application and configure metadata
app = FastAPI(
    title="D2C Customer Churn Prediction Service",
    description="Internal REST API providing real-time and batch customer attrition risk scoring.",
    version="1.0.0"
)

# Global variables to store the model and scaler artifacts
model = None
scaler = None
MODEL_PATH = "./model.pkl"

RAW_MODEL_URL = "https://raw.githubusercontent.com/DaivaPepalla/Part3-Churn_Prediction_Model-Model_Card/main/model.pkl"


# =================================================================
#  SELF-SUSTAINING LIFECYCLE EVENT HANDLER
# =================================================================

class CustomerFeatures(BaseModel):
    customer_id: str = Field(..., example="CUST_9948")
    recency: int = Field(..., ge=0, example=24, description="Days since last successful purchase transaction.")
    frequency: int = Field(..., ge=0, example=6, description="Total cumulative transaction count.")
    monetary_value: float = Field(..., ge=0.0, example=4250.50, description="Total historical spend footprint.")
    support_complaints: int = Field(..., ge=0, example=1, description="Total opened customer help desk tickets.")
    web_interactions: int = Field(..., ge=0, example=18, description="Cumulative app/web login activity density.")

class PredictionResponse(BaseModel):
    customer_id: str
    churn_probability: float
    predicted_class: int
    risk_explanation: str

class BatchPredictionRequest(BaseModel):
    customers: List[CustomerFeatures]

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]

# =================================================================
#  LIFECYCLE EVENT HANDLER
# =================================================================
@app.on_event("startup")
def load_artifacts():
    global model, scaler
    if not os.path.exists(MODEL_PATH):
        print(f"📡 'model.pkl' not found locally. Fetching directly from Part 3 GitHub Repository...")
        try:
            # Using urllib to download the binary stream cleanly without external dependencies like requests
            urllib.request.urlretrieve(RAW_MODEL_URL, MODEL_PATH)
            print(" Successfully pulled and cached model.pkl from Part 3 GitHub!")
        except Exception as e:
            raise RuntimeError(
                f" Critical Error: Failed to auto-download model.pkl from GitHub remote source. "
                f"Please ensure the file is public in your Part 3 repository. Details: {e}"
            )
    try:
        with open(MODEL_PATH, "rb") as f:
            artifacts = pickle.load(f)
            
            # Handle flexibility if model is raw or packed in a dictionary pipeline
            if isinstance(artifacts, dict):
                model = artifacts.get("model")
                scaler = artifacts.get("scaler")
            else:
                model = artifacts
                scaler = None
        print(" Supervised model and normalization pipelines successfully loaded into memory!")
    except Exception as e:
        raise RuntimeError(f" Failed to parse or deserialize model file: {str(e)}")

def generate_risk_explanation(features: CustomerFeatures, prob: float) -> str:
    """Generates plain-English structural risk drivers for non-technical stakeholders."""
    if prob < 0.35:
        return "Account is stable. Behavioral density patterns match retention profiles."
    
    reasons = []
    if features.support_complaints >= 3:
        reasons.append("high support desk friction")
    if features.recency > 45:
        reasons.append("extended transactional silence")
    if features.web_interactions < 5:
        reasons.append("dropping digital interaction footprints")
        
    if not reasons:
        reasons.append("marginal behavioral pattern drift")
        
    return f"High Risk Cutoff Breached. Attrition driven primary by: {', '.join(reasons)}."

# =================================================================
#  API ENDPOINT ROUTING MATRIX
# =================================================================
@app.get("/health", status_code=200)
def health_check():
    """Validates structural state parameters for automated container orchestrators."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "environment": os.getenv("ENV", "production")
    }

@app.post("/predict", response_model=PredictionResponse)
def predict_single(payload: CustomerFeatures):
    """Calculates immediate real-time churn probabilities for an individual customer payload."""
    try:
        # Extract features array ordered identically to model training sequence
        raw_vector = np.array([[
            payload.recency,
            payload.frequency,
            payload.monetary_value,
            payload.support_complaints,
            payload.web_interactions
        ]])
        
        # Apply standard scale normalization pipeline if present
        processed_vector = scaler.transform(raw_vector) if scaler else raw_vector
        
        # Calculate positive class probability matrix
        prob = float(model.predict_proba(processed_vector)[0][1])
        
        # Apply optimized operational business decision threshold
        pred_class = 1 if prob >= 0.35 else 0
        explanation = generate_risk_explanation(payload, prob)
        
        return PredictionResponse(
            customer_id=payload.customer_id,
            churn_probability=round(prob, 4),
            predicted_class=pred_class,
            risk_explanation=explanation
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal prediction engine failure: {str(e)}")

@app.post("/batch_predict", response_model=BatchPredictionResponse)
def predict_batch(payload: BatchPredictionRequest):
    """Processes large arrays of customer feature profiles in a single vectorized pass."""
    try:
        results = []
        raw_matrix = []
        
        for p in payload.customers:
            raw_matrix.append([p.recency, p.frequency, p.monetary_value, p.support_complaints, p.web_interactions])
            
        # Execute fast multi-row operations across input arrays
        np_matrix = np.array(raw_matrix)
        processed_matrix = scaler.transform(np_matrix) if scaler else np_matrix
        probabilities = model.predict_proba(processed_matrix)[:, 1]
        
        for idx, p in enumerate(payload.customers):
            prob = float(probabilities[idx])
            pred_class = 1 if prob >= 0.35 else 0
            explanation = generate_risk_explanation(p, prob)
            
            results.append(PredictionResponse(
                customer_id=p.customer_id,
                churn_probability=round(prob, 4),
                predicted_class=pred_class,
                risk_explanation=explanation
            ))
            
        return BatchPredictionResponse(predictions=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal batch prediction pipeline failure: {str(e)}")