import pandas as pd
from sklearn.model_selection import train_test_split

# Function to prepare dataset
def prepare_dataset(dataset: pd.DataFrame):
	x_df = dataset.drop(labels=["LABEL"], axis=1)
	y_df = dataset["LABEL"]

	return x_df, y_df

# Function to split dataset
def split_dataset(x_df, y_df):
	X_train, X_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.2, random_state=42)

	return X_train, X_test, y_train, y_test
