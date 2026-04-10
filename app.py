from flask import Flask, request, jsonify, send_from_directory
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix, roc_curve)
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, static_folder=BASE_DIR, static_url_path='')

@app.after_request
def add_cors(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    return response

def generate_dataset(n=2000, seed=42):
    rng = np.random.default_rng(seed)
    tenure          = rng.integers(1, 72, n)
    monthly_charges = rng.uniform(20, 120, n)
    total_charges   = tenure * monthly_charges + rng.normal(0, 50, n)
    num_services    = rng.integers(1, 8, n)
    contract_type   = rng.choice([0, 1, 2], n, p=[0.5, 0.25, 0.25])
    payment_method  = rng.integers(0, 4, n)
    support_calls   = rng.integers(0, 10, n)
    age_group       = rng.integers(0, 3, n)
    has_partner     = rng.integers(0, 2, n)
    has_dependents  = rng.integers(0, 2, n)
    churn_prob = (
        0.4 * (1 - tenure / 72) +
        0.25 * (monthly_charges / 120) +
        0.15 * (support_calls / 10) +
        0.10 * (1 - contract_type / 2) +
        0.05 * (1 - num_services / 7) +
        0.05 * rng.random(n)
    )
    churn = (churn_prob > 0.45).astype(int)
    return pd.DataFrame({
        'tenure': tenure, 'monthly_charges': monthly_charges.round(2),
        'total_charges': np.clip(total_charges, 0, None).round(2),
        'num_services': num_services, 'contract_type': contract_type,
        'payment_method': payment_method, 'support_calls': support_calls,
        'age_group': age_group, 'has_partner': has_partner,
        'has_dependents': has_dependents, 'churn': churn
    })

def train_model():
    df = generate_dataset()
    X = df.drop('churn', axis=1); y = df['churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train); X_test_s = scaler.transform(X_test)
    gb  = GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=5, subsample=0.8, random_state=42)
    rf  = RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_split=5, random_state=42, n_jobs=-1)
    ensemble = VotingClassifier(estimators=[('gb', gb), ('rf', rf)], voting='soft', weights=[2, 1])
    ensemble.fit(X_train_s, y_train)
    y_pred = ensemble.predict(X_test_s); y_prob = ensemble.predict_proba(X_test_s)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob); cm = confusion_matrix(y_test, y_pred)
    metrics = {
        'accuracy' : round(accuracy_score(y_test, y_pred)*100, 2),
        'precision': round(precision_score(y_test, y_pred)*100, 2),
        'recall'   : round(recall_score(y_test, y_pred)*100, 2),
        'f1'       : round(f1_score(y_test, y_pred)*100, 2),
        'roc_auc'  : round(roc_auc_score(y_test, y_prob)*100, 2),
        'confusion_matrix': cm.tolist(),
        'roc_curve': {'fpr':[round(f,4) for f in fpr.tolist()], 'tpr':[round(t,4) for t in tpr.tolist()]},
        'churn_rate': round(y.mean()*100, 2), 'dataset_size': len(df),
        'feature_importance': dict(zip(X.columns.tolist(),
            [round(v,4) for v in ensemble.estimators_[0].feature_importances_.tolist()]))
    }
    cv = cross_val_score(ensemble, scaler.transform(X), y, cv=5, scoring='roc_auc')
    metrics['cv_roc_auc'] = round(cv.mean()*100, 2)
    return ensemble, scaler, X.columns.tolist(), metrics

print("Training ensemble model...")
MODEL, SCALER, FEATURES, TRAIN_METRICS = train_model()
print("Model ready!")

@app.route('/')
def index():
    return send_from_directory(BASE_DIR, 'index.html')

@app.route('/api/metrics')
def metrics():
    return jsonify(TRAIN_METRICS)

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json()
    try:
        row = pd.DataFrame([{
            'tenure': int(data['tenure']), 'monthly_charges': float(data['monthly_charges']),
            'total_charges': float(data['total_charges']), 'num_services': int(data['num_services']),
            'contract_type': int(data['contract_type']), 'payment_method': int(data['payment_method']),
            'support_calls': int(data['support_calls']), 'age_group': int(data['age_group']),
            'has_partner': int(data['has_partner']), 'has_dependents': int(data['has_dependents']),
        }])
        prob = MODEL.predict_proba(SCALER.transform(row))[0][1]
        risk_level = 'High' if prob >= 0.7 else ('Medium' if prob >= 0.4 else 'Low')
        suggestions = []
        if int(data['contract_type']) == 0: suggestions.append("Offer a 1-year or 2-year contract discount")
        if float(data['monthly_charges']) > 80: suggestions.append("Consider a loyalty pricing plan")
        if int(data['support_calls']) > 5: suggestions.append("Escalate to a dedicated support agent")
        if int(data['num_services']) < 3: suggestions.append("Bundle additional services with incentives")
        if int(data['tenure']) < 12: suggestions.append("Enroll in new-customer retention program")
        return jsonify({'churn': int(prob>=0.5), 'probability': round(prob*100,2),
                        'risk_level': risk_level,
                        'suggestions': suggestions or ["Customer appears stable — maintain engagement"]})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/bulk_predict', methods=['POST'])
def bulk_predict():
    rng = np.random.default_rng(); n = int(request.get_json().get('n', 20))
    rows = [{'tenure':int(rng.integers(1,72)),'monthly_charges':float(round(rng.uniform(20,120),2)),
              'total_charges':float(round(rng.uniform(100,8000),2)),'num_services':int(rng.integers(1,8)),
              'contract_type':int(rng.choice([0,1,2])),'payment_method':int(rng.integers(0,4)),
              'support_calls':int(rng.integers(0,10)),'age_group':int(rng.integers(0,3)),
              'has_partner':int(rng.integers(0,2)),'has_dependents':int(rng.integers(0,2))} for _ in range(n)]
    df = pd.DataFrame(rows)
    probs = MODEL.predict_proba(SCALER.transform(df))[:,1]
    df['churn_probability'] = (probs*100).round(2)
    df['risk'] = pd.cut(probs, bins=[0,.4,.7,1], labels=['Low','Medium','High'])
    return jsonify(df.to_dict(orient='records'))

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
