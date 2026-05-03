from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

def plot_confution_matrix(cm, title, save_path):
	Path(save_path).parent.mkdir(parents=True, exist_ok=True)

	disp = ConfusionMatrixDisplay(
		confusion_matrix=cm,
		display_labels=["REAL", "FAKE"]
	)

	disp.plot(values_format="d")

	plt.title(title)
	plt.tight_layout()
	plt.savefig(save_path, dpi=300)
	plt.close()

def plot_roc_curve(y_true, y_score, title, save_path):
	Path(save_path).parent.mkdir(parents=True, exist_ok=True)

	RocCurveDisplay.from_predictions(y_true, y_score)
	plt.title(title)
	plt.tight_layout()
	plt.savefig(save_path, dpi=300)
	plt.close()

def plot_bar(results_df, metric, title, ylabel, save_path):
	Path(save_path).parent.mkdir(parents=True, exist_ok=True)

	plt.figure(figsize=(8, 5))
	plt.bar(results_df["model"], results_df["metric"])
	plt.title(title)
	plt.ylabel(ylabel)
	plt.xlabel("Model")
	plt.xticks(rotation=30, ha="right")
	plt.tight_layout()
	plt.savefig(save_path, dpi=300)
	plt.close()

