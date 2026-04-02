"""Train a fraud classifier using the project's synthetic dataset.

Saves a scikit-learn RandomForestClassifier to `artifacts/fraud_classifier.joblib`.
"""
import os
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import joblib

from .utils import load_data, preprocess


ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), 'artifacts')
os.makedirs(ARTIFACT_DIR, exist_ok=True)
MODEL_PATH = os.path.join(ARTIFACT_DIR, 'fraud_classifier.joblib')
METRICS_PATH = os.path.join(ARTIFACT_DIR, 'fraud_metrics.json')


def train_and_save(data_path=None, random_state=42):
    df = load_data(data_path)
    X, y = preprocess(df)
    # handle class imbalance with class_weight
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y)

    clf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=random_state, n_jobs=-1, class_weight='balanced')
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    metrics = {'classification_report': classification_report(y_test, preds, output_dict=True)}
    try:
        proba = clf.predict_proba(X_test)[:, 1]
        metrics['roc_auc'] = float(roc_auc_score(y_test, proba))
    except Exception:
        metrics['roc_auc'] = None

    # Save model + metrics
    joblib.dump(clf, MODEL_PATH)
    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics, f, indent=2)

    print('Model saved to', MODEL_PATH)
    print('Metrics saved to', METRICS_PATH)
    return MODEL_PATH, METRICS_PATH


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train fraud classifier')
    parser.add_argument('--data', type=str, default=None, help='Path to CSV data')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    train_and_save(data_path=args.data, random_state=args.seed)
