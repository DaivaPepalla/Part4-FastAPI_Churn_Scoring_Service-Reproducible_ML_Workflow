# Churn API Post-Deployment Monitoring & Responsible Use Plan

Once this FastAPI microservice is deployed to production, a continuous monitoring framework and ethical guidelines must be established to maintain long-term model reliability and fair operational practices.

---

## 📈 Post-Deployment Monitoring Plan

### 1. Model & Data Drift Tracking
* **Data Drift:** Monitor incoming inference features (e.g., `recency`, `support_complaints`) against the training baseline distributions using metrics like **Population Stability Index (PSI)** or the **Kolmogorov-Smirnov test**. Significant shifts indicate changing customer behavior patterns.
* **Prediction Drift:** Track the distribution of output probabilities over time. If the average `churn_probability` or the ratio of `predicted_class: 1` spikes significantly without an operational explanation, the model's decision boundaries may be losing alignment.

### 2. Software & API Performance Metrics
* **Error Rates:** Track the volume of HTTP status codes. A spike in `422 Unprocessable Entity` indicates upstream data pipeline mismatches or broken payloads; a spike in `500 Internal Server Error` indicates operational bugs or infrastructure failure.
* **Latency:** Monitor response times for both `/predict` (target < 50ms) and `/batch_predict` (target < 200ms per batch array block) to ensure smooth integration with CRM systems.

### 3. Business Outcomes & Feedback Loops
* **True Retention vs Attrition:** Cross-reference predictions with actual customer outcomes 30 to 60 days post-inference to calculate production accuracy, precision, and recall matrices.
* **Intervention ROI:** Track whether accounts flagged as high-risk (`predicted_class: 1`) have higher save-rates when a customer success representative intervenes versus a control group.

### 4. Automated Retraining Triggers
Schedule a pipeline recalculation event if any of the following boundaries are crossed:
* A sustained drop in model precision or recall below a set baseline (e.g., dropping below 75% accuracy).
* Population Stability Index (PSI) on core metrics exceeding `0.2`.
* A fixed maximum time window elapsed (e.g., proactive automated retraining every 90 days to capture seasonal market fluctuations).

---

## ⚖️ Responsible Use Guidelines for the Retention Team

To ensure ethical, fair, and high-ROI operational deployment, the business consulting and retention teams must adhere to the following usage guidelines:

### ✅ How the API Output SHOULD Be Used
* **Proactive Outreach:** Use high-probability flags to prioritize customer success check-ins, offering personalized engagement audits, feature education, or value optimization.
* **Loyalty Incentives:** Direct marketing budgets toward high-value, high-risk profiles by extending tailored loyalty rewards, service discounts, or early renewal options.
* **Systemic Friction Resolution:** Look at aggregated batch prediction risk explanations to identify systemic product bottlenecks (e.g., if many accounts show risk due to "high support complaints," flag that product module to the engineering team).

### ❌ How the API Output SHOULD NOT Be Used
* **Punitive Pricing or Penalties:** Do not use high churn risk predictions to prematurely penalize customers, raise subscription fees, restrict account features, or implement aggressive contractual lock-ins.
* **Automated Account Termination:** The model must never be used to automatically offboard or deprioritize accounts without human-in-the-loop validation from an account manager.
* **Bias and Discrimination:** Ensure intervention workflows do not create systematic disparities across demographic segments or regions not explicitly designed into the behavioral feature matrix.