# Anomaly Detection (export)

This repository contains an exported subset of the AuditAI project focused on anomaly (fraud) detection. It is intended as a minimal, portable package with the training notebook, training script, dataset, and a pre-trained model artifact so others can reproduce the experiments quickly.

Contents
- `anomaly-pack/`
  - `notebooks/` — `model_experiments.ipynb`, `train_models.py`, `datagen.py`, `requirements.txt`
  - `fraud_model/` — `model.py`, `train.py`, `utils.py`, and `artifacts/` (includes `fraud_classifier.joblib` and `fraud_metrics.json`)
  - `data/` — `synthetic_enterprise_audit.csv` (synthetic dataset used for experiments)

Quickstart
1. Clone the repository:

```bash
git clone https://github.com/Alihasnat930/anomaly-detection.git
cd anomaly-detection/anomaly-pack
```

2. Create a Python virtual environment and install dependencies:

```bash
python -m venv venv
# Unix/macOS
source venv/bin/activate
# Windows PowerShell
venv\Scripts\Activate.ps1
pip install -r notebooks/requirements.txt
```

3. Run the notebook or training script:

```bash
# Run the notebook interactively (Jupyter)
jupyter notebook notebooks/model_experiments.ipynb

# Or run the training script directly
python notebooks/train_models.py
```

4. Load the trained model in Python:

```python
from fraud_model.model import FraudModel
m = FraudModel.load('fraud_model/artifacts/fraud_classifier.joblib')
# prepare features using the notebook's build_feature_matrix
preds = m.predict_proba(X)[:, 1]
```

Notes
- The dataset `data/synthetic_enterprise_audit.csv` is synthetic and included for demonstration. It is ~18MB in size.
- The model artifact `fraud_classifier.joblib` is included (~2MB). If you prefer not to keep large artifacts in the repository, consider using Git LFS.

License & attribution
- Files were exported from the AuditAI project for sharing the anomaly detection experiments and model.

If you'd like, I can (a) add a LICENSE file, (b) split the dataset into a smaller sample, or (c) convert the model artifact to ONNX and add inference code. Which should I do next?
