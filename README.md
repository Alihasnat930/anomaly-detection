# Anomaly Detection — Example Package

This repository is a standalone example package demonstrating anomaly (fraud) detection on transaction data. It includes a synthetic dataset, training notebooks and scripts, and a pre-trained classifier to enable quick reproduction of the training and evaluation workflow.

Contents
- `anomaly-pack/`
  - `notebooks/` — `model_experiments.ipynb`, `train_models.py`, `datagen.py`, `requirements.txt`
  - `fraud_model/` — `model.py`, `train.py`, `utils.py`, and `artifacts/` (contains `fraud_classifier.joblib` and `fraud_metrics.json`)
  - `data/` — `synthetic_enterprise_audit.csv` (synthetic dataset used for experiments)

Quickstart
1. Clone the repository and open the `anomaly-pack` folder:

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

3. Run the notebook or the training script:

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
- The dataset `data/synthetic_enterprise_audit.csv` is synthetic and included for demonstration (approx. 18MB).
- The model artifact `fraud_classifier.joblib` is included (approx. 2MB). If you prefer not to keep large artifacts in the repository, consider using Git LFS.

License
- Check or add a suitable open-source license to clarify reuse (e.g., MIT).

Next steps
- I can add a `LICENSE`, create a smaller sample dataset, or add inference examples (ONNX conversion) on request.
