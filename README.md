# 🛡️ ChurnShield — Customer Churn Prediction

A fully local, no-API-key customer churn prediction platform.

## Stack
- **Backend** — Python · Flask · scikit-learn Voting Ensemble (Gradient Boosting + Random Forest)
- **Frontend** — Animated dark-sci-fi HTML/CSS/JS with Chart.js

---

## ⚡ Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the server
```bash
python app.py
```

### 3. Open the app
Visit **http://127.0.0.1:5000** in your browser.

---

## Features

| Feature | Details |
|---|---|
| **Dashboard** | Accuracy, ROC-AUC, F1, Recall, Confusion Matrix, Feature Importance |
| **Predict** | Real-time single-customer churn risk + actionable retention suggestions |
| **Bulk Analysis** | Batch score N random customers, visualize distribution, risk table |

## ML Model
- **Gradient Boosting** (300 estimators, lr=0.05, depth=5) — weight 2
- **Random Forest** (200 estimators, depth=8) — weight 1
- Soft-voting ensemble
- 5-fold cross-validated ROC-AUC reported

---
No API keys. No external services. Runs entirely on your machine.
