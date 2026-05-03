import os
import kagglehub
import pandas as pd

def load_dataset():
	# Download latest version
	path = kagglehub.dataset_download("birdy654/deep-voice-deepfake-voice-recognition")

	# Read dataset from path
	df = pd.read_csv(os.path.join(path, "DATASET-balanced.csv"))

	#
	return df


