"""Wrapper for the trained fraud classifier.

Provides a simple interface to load a trained artifact and predict on
preprocessed feature matrices or pandas DataFrames.
"""
import os
import joblib
import numpy as np
import pandas as pd


DEFAULT_ARTIFACT = os.path.join(os.path.dirname(__file__), 'artifacts', 'fraud_classifier.joblib')


class FraudModel:
    def __init__(self, model):
        self.model = model

    @classmethod
    def load(cls, path=None):
        path = path or DEFAULT_ARTIFACT
        model = joblib.load(path)
        return cls(model)

    def predict(self, X):
        """Return binary predictions for X (array or DataFrame)."""
        arr = self._ensure_array(X)
        return self.model.predict(arr)

    def predict_proba(self, X):
        arr = self._ensure_array(X)
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(arr)
        raise AttributeError('Underlying model has no predict_proba')

    def _ensure_array(self, X):
        if isinstance(X, pd.DataFrame):
            return X.values
        if isinstance(X, (list, tuple)):
            return np.array(X)
        return X
