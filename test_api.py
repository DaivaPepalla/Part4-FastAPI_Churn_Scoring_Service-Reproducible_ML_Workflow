import os
import pickle
import pytest
from fastapi.testclient import TestClient
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

# Setup automated dummy model fallback if script runs standalone
if not os.path.exists("./model.pkl"):
    print("🔧 Generating dummy testing model configuration file...")
    X, y = make_classification(n_samples=20, n_features=5, random_state=42)
    dummy_model = RandomForestClassifier(random_state=42).fit(X, y)
    with open("./model.pkl", "wb") as f:
        pickle.dump({"model": dummy_model, "scaler": None}, f)

from app.main import app

client = TestClient(app)

def test_01_health_endpoint():
    """Verify health systems state parameters read correctly."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] is True

def test_02_predict_single_success():
    """Test standard single customer payload handling at optimized threshold boundaries."""
    payload = {
        "customer_id": "CUST_TEST_01",
        "recency": 12,
        "frequency": 8,
        "monetary_value": 3500.0,
        "support_complaints": 0,
        "web_interactions": 24
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["customer_id"] == "CUST_TEST_01"
    assert "churn_probability" in data
    assert data["predicted_class"] in [0, 1]
    assert isinstance(data["risk_explanation"], str)

def test_03_predict_validation_failure():
    """Verify input filters block corrupt parameters (e.g., negative complaints count)."""
    corrupt_payload = {
        "customer_id": "CUST_ERR_01",
        "recency": 12,
        "frequency": 8,
        "monetary_value": 3500.0,
        "support_complaints": -5,  # Invalid configuration bounds
        "web_interactions": 24
    }
    response = client.post("/predict", json=corrupt_payload)
    assert response.status_code == 422  # Unprocessable Entity
    assert "detail" in response.json()

def test_04_batch_prediction():
    """Verify bulk array vectors process safely inside a single batch pass."""
    batch_payload = {
        "customers": [
            {"customer_id": "BCUST_1", "recency": 5, "frequency": 12, "monetary_value": 8200.0, "support_complaints": 0, "web_interactions": 45},
            {"customer_id": "BCUST_2", "recency": 85, "frequency": 1, "monetary_value": 450.0, "support_complaints": 4, "web_interactions": 2}
        ]
    }
    response = client.post("/batch_predict", json=batch_payload)
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert len(data["predictions"]) == 2
    assert data["predictions"][1]["customer_id"] == "BCUST_2"

if __name__ == "__main__":
    pytest.main(["-v"])