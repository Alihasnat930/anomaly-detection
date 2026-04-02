"""Utilities for training and preprocessing for the fraud model.

Contains helpers to load the synthetic CSV and to assemble the feature matrix
used by the backend fraud training/prediction code.
"""
import os
import pandas as pd
import numpy as np


def data_path():
	# default dataset location (project root/data/processed)
	root = os.path.dirname(
		os.path.dirname(
			os.path.dirname(
				os.path.dirname(
					os.path.dirname(__file__)
				)
			)
		)
	)
	return os.path.join(root, 'data', 'processed', 'synthetic_enterprise_audit.csv')


def load_data(path=None):
	path = path or data_path()
	return pd.read_csv(path)


def preprocess(df):
	"""Basic preprocessing returning feature DataFrame X and label y.

	This mirrors the notebook/script preprocessing: fills missing values,
	one-hot encodes small categorical columns and returns a numeric matrix.
	"""
	df = df.copy()

	FEATURES = ['amount', 'tax_amount', 'vendor_risk_score', 'user_tenure_days', 'previous_txn_count']
	for col in ['payment_method', 'category', 'approval_status']:
		if col in df.columns:
			df[col] = df[col].fillna('Unknown')
			dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
			df = pd.concat([df, dummies], axis=1)

	for f in FEATURES:
		if f in df.columns:
			df[f] = df[f].fillna(df[f].median())

	selected_cols = [c for c in df.columns if c in FEATURES or c.startswith('payment_method_') or c.startswith('category_') or c.startswith('approval_status_')]
	X = df[selected_cols]
	y = df['anomaly_flag'] if 'anomaly_flag' in df.columns else None
	return X, y
