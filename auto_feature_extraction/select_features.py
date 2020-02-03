import pandas as pd
from tsfresh import extract_features, select_features
from os import listdir
from os.path import isfile, join
from auto_feature_extraction.config import *


def select():
	features_files = [f for f in listdir(features_dir) if isfile(join(features_dir, f))]

	# Select features individually from each of the signal components
	train = pd.DataFrame()
	test = pd.DataFrame()
	for f_file in features_files:  # One file for each signal component

		print(f"loading {f_file}")
		features = pd.read_csv(features_dir + f_file)

		train_x = features.iloc[:validation_split_i].drop('y', axis=1)
		test_x = features.iloc[validation_split_i:].drop('y', axis=1)
		train_y = features.iloc[:validation_split_i].y
		test_y = features.iloc[validation_split_i:].y

		# Feature selection must be always done from the train set!
		print("selecting features...")
		train_features_selected = select_features(train_x, train_y, fdr_level=fdr_level)

		print(f"selected {len( train_features_selected.columns )} features.")

		comp_train = train_features_selected.copy()
		comp_test = test_x[train_features_selected.columns].copy()

		train = pd.concat([train, comp_train], axis=1)
		test = pd.concat([test, comp_test], axis=1)

	train['y'] = train_y
	test['y'] = test_y

	print(f"saving {train_file}")
	train.to_csv(train_file, index=None)

	print(f"saving {test_file}")
	test.to_csv(test_file, index=None)


if __name__ == '__main__':
	select()
