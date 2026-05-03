import numpy as np
from sklearn.metrics import (
	accuracy_score,
	precision_score,
	recall_score,
	f1_score,
	roc_auc_score,
	roc_curve,
	confusion_matrix
)

# Function to compute eer (Equal Error Rate)
def compute_eer(y_true, y_pred):
	fpr, tpr, threshold = roc_curve(y_true, y_pred)

	fnr = 1 - tpr

	idx = np.nanargmin(np.abs(fpr - fnr))
	eer = (fpr[idx] + fnr[idx]) / 2.0

	return eer * 100.0

# Function to compute basic metrics
def compute_basic_metrics(y_true, y_pred):
	return {
		"accuracy": accuracy_score(y_true, y_pred) * 100.0,
		"precision": precision_score(y_true, y_pred) * 100.0,
		"recall": recall_score(y_true, y_pred) * 100.0,
		"f1": f1_score(y_true, y_pred) * 100.0,
		"auc": roc_auc_score(y_true, y_pred) * 100.0,
		"eer": compute_eer(y_true, y_pred)
	}

# Compute false positive rate
def compute_home_false_positive_rate(y_home_pred):
	return np.mean(y_home_pred == 1) * 100.0

# Compute Confusion Matrix
def confusion(y_true, y_pred):
	return confusion_matrix(y_true, y_pred, labels=[0, 1])