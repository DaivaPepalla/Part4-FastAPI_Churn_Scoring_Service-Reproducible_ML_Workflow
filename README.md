# Churn Prediction Model - API Deployment (Part 4)

This repository contains the production-ready **FastAPI** microservice gateway for the Churn Prediction Model. The application securely handles incoming customer metrics, validates schemas using Pydantic, fetches the latest trained model artifact directly from the upstream repository, and serves real-time single and batch prediction streams.

---

## 🛠️ System Architecture
```text
D:\Capstone project\Part-4\
├── app/
│   └── main.py
├── data/
│   └── csv files
├── tests/
│   └── test_api.py
├── model.pkl
├── monitoring_plan.md
├── README.md
└── requirements.txt
```
The gateway is built for high-throughput operational stability:
* **Dynamic Artifact Resolution:** On startup, the service securely downloads the verified model pipeline binary (`model.pkl`) directly from the Part 3 cloud repository CDN, ensuring complete environmental synchronization.
* **Strict Pydantic Contract Layer:** Input features are passed through explicit structural validation types to handle anomalies before they touch the predictive engine.
* **Scikit-Learn Synchronization:** Optimized to run seamlessly against `scikit-learn==1.4.1.post1`.

---

## 🚀 Getting Started & Installation

### 1. Prerequisites
Ensure your global environment is synchronized with the core runtime requirements:
* Python 3.10.x
* Git Bash (or standard Windows Command Prompt)

### 2. Environment Verification
Ensure your local `site-packages` clean room matches the baseline runtime signature exactly:
```bash
python -m pip install scikit-learn==1.4.1.post1 uvicorn fastapi pydantic requests
3. Launching the Gateway Server
Navigate to the root directory of your Part 4 workspace and initialize the Uvicorn engine:
# Using standard Windows CMD
C:\Python310\python.exe -m uvicorn app.main:app --reload

# Using Git Bash Terminal
/c/Python310/python.exe -m uvicorn app.main:app --reload
Once initialized, the live interactive Swagger UI Documentation portal is accessible via your web browser at:
👉 http://127.0.0.1:8000/docs

📡 API Endpoint Matrix & Specifications
🟢 1. System Health Engine [GET /]
Verifies application uptime, network operational sanity, and confirms that the model binary is properly mounted in memory.

Sample Response:
{
  "status": "healthy",
  "model_loaded": true,
  "environment": "production"
}
🟢 2. Single Account Scoring [POST /predict]
Scores an individual customer footprint using a custom machine learning payload matrix.

Required Data Schema (JSON):
{
  "customer_id": "CUST_8821",
  "recency": 45,
  "frequency": 2,
  "monetary_value": 120.50,
  "support_complaints": 4,
  "web_interactions": 1
}
Sample Output Payload:

JSON
{
  "customer_id": "CUST_8821",
  "churn_probability": 0.5441,
  "predicted_class": 1,
  "risk_explanation": "High Risk Cutoff Breached. Attrition driven primary by: high support desk friction, dropping digital interaction footprints."
}
🟢 3. Enterprise Bulk Processor [POST /batch_predict]
Processes collections of multiple customer records simultaneously for high-throughput operational scaling.

Required Data Schema (JSON):

JSON
{
  "customers": [
    {
      "customer_id": "CUST_BATCH_01",
      "recency": 45,
      "frequency": 2,
      "monetary_value": 120.50,
      "support_complaints": 4,
      "web_interactions": 1
    },
    {
      "customer_id": "CUST_BATCH_02",
      "recency": 2,
      "frequency": 28,
      "monetary_value": 450.00,
      "support_complaints": 0,
      "web_interactions": 35
    }
  ]
}
📦 Core Pipeline Validations
The application features a strict verification suite to intercept operational bugs early:

Schema Enforcement: Missing attributes or invalid metric tracking data formats immediately throw a localized 422 Unprocessable Entity response, safeguarding downstream model matrices from corruption.

Version Isolation: Automatically blocks binary drift conflicts between training configurations and local server package layers..
