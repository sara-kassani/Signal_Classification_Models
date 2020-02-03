import warnings
import numpy as np
import pandas as pd
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from auto_feature_extraction.config import features_dir
from data_handling import load_dataset


def extract():
	X_train, X_test, y_train, y_test = load_dataset(verbose=0)

	# Reshape the data to the right input format for tfresh
	X = np.concatenate([X_train, X_test])
	y = np.concatenate([y_train, y_test])

	# Extract the features from each of the signal components
	for sig_comp in range(X.shape[2]):
		signal = X[:, :, sig_comp]
		d = pd.DataFrame(signal)

		d = d.stack()
		d.index.rename(['id', 'time'], inplace=True)
		d = d.reset_index()

		# TODO fix this, the warnings keep showing
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			f = extract_features(d, column_id="id", column_sort="time")

		impute(f)
		assert f.isnull().sum().sum() == 0

		print(f"Features extracted from sig_comp_{sig_comp}: {f.shape[1]}")

		f['y'] = y
		# File where to save all the extracted features
		filename = f"signal_comp_{sig_comp}.csv"
		f.to_csv(features_dir + filename, index=None)


if __name__ == '__main__':
	extract()
