"""
Runnable script to load synthetic dataset, run basic EDA, train two baseline models,
and save them to `notebooks/artifacts/`.
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, roc_auc_score, mean_squared_error
import joblib

# Settings
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'processed', 'synthetic_enterprise_audit.csv')
ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), 'artifacts')
os.makedirs(ARTIFACT_DIR, exist_ok=True)

print('Loading', DATA_PATH)
df = pd.read_csv(DATA_PATH)
print('rows,cols:', df.shape)

# Simple EDA summaries
print(df.dtypes)
print('\nMissing values:\n', df.isna().sum())
print('\nAnomaly rate:', df['anomaly_flag'].mean())

# Preprocessing
FEATURES = ['amount','tax_amount','vendor_risk_score','user_tenure_days','previous_txn_count']
# One-hot encode categories we expect
for col in ['payment_method','category','approval_status']:
    if col in df.columns:
        df[col] = df[col].fillna('Unknown')
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
        df = pd.concat([df, dummies], axis=1)

for f in FEATURES:
    if f in df.columns:
        df[f] = df[f].fillna(df[f].median())

# Build feature matrix
selected_cols = [c for c in df.columns if c in FEATURES or c.startswith('payment_method_') or c.startswith('category_') or c.startswith('approval_status_')]
X = df[selected_cols]
y_fraud = df['anomaly_flag']
y_risk = df['risk_score']

# Train/test split
X_train, X_test, yf_train, yf_test = train_test_split(X, y_fraud, test_size=0.2, random_state=42, stratify=y_fraud)
_, _, yr_train, yr_test = train_test_split(X, y_risk, test_size=0.2, random_state=42)

# Fraud classifier
clf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
clf.fit(X_train, yf_train)
yf_pred = clf.predict(X_test)
print('\nFraud classifier report:\n')
print(classification_report(yf_test, yf_pred))
if hasattr(clf, 'predict_proba'):
    try:
        proba = clf.predict_proba(X_test)[:,1]
        print('ROC AUC:', roc_auc_score(yf_test, proba))
    except Exception:
        pass
joblib.dump(clf, os.path.join(ARTIFACT_DIR, 'fraud_classifier.joblib'))

# Risk regressor
rgr = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
rgr.fit(X_train, yr_train)
yr_pred = rgr.predict(X_test)
print('\nRisk regressor RMSE:', mean_squared_error(yr_test, yr_pred, squared=False))
joblib.dump(rgr, os.path.join(ARTIFACT_DIR, 'risk_regressor.joblib'))

print('Artifacts saved to', ARTIFACT_DIR)
