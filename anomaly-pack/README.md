# Anomaly Pack

This folder contains the materials for training and evaluating an anomaly (fraud) detection model on synthetic transaction data. It is organized to make it easy to reproduce experiments and to load the trained classifier for demonstration.

Structure
- `notebooks/` — EDA, feature engineering, training, and evaluation notebooks and scripts.
- `fraud_model/` — lightweight model wrapper, training helper, and utilities. Includes `artifacts/` with a pre-trained `fraud_classifier.joblib` and `fraud_metrics.json`.
- `data/` — `synthetic_enterprise_audit.csv` (synthetic dataset used by the notebooks and scripts).

Quickstart
1. From the repository root, change into this folder:

```bash
cd anomaly-pack
```

2. Create and activate a virtual environment, then install dependencies:

```bash
python -m venv venv
# Unix/macOS
source venv/bin/activate
# Windows PowerShell
venv\Scripts\Activate.ps1
pip install -r notebooks/requirements.txt
```

3. Reproduce experiments:

```bash
jupyter notebook notebooks/model_experiments.ipynb
# or run the training script
python notebooks/train_models.py
```

Loading the model
```python
from fraud_model.model import FraudModel
m = FraudModel.load('fraud_model/artifacts/fraud_classifier.joblib')
preds = m.predict_proba(X)[:, 1]
```

Notes
- The dataset is synthetic and intended for demonstration and testing.
- Consider using Git LFS for large artifacts/datasets if you plan to keep them in the repository.

If you want, I can add a small example script that shows a complete end-to-end inference flow from CSV -> features -> model predictions.
