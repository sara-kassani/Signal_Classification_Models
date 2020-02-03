import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score as accuracy
from auto_feature_extraction.config import train_file, test_file
from sklearn.neural_network import MLPClassifier


def train_evaluate():
	train = pd.read_csv(train_file)
	test = pd.read_csv(test_file)

	x_train = train.drop('y', axis=1).values
	y_train = train.y.values
	x_test = test.drop('y', axis=1).values
	y_test = test.y.values

	classifiers = [
		make_pipeline(MinMaxScaler(), LogisticRegression(max_iter=300)),
		make_pipeline(StandardScaler(), LogisticRegression(C=30, max_iter=300)),
		make_pipeline(MinMaxScaler(), SVC(kernel='rbf')),
		make_pipeline(MinMaxScaler(), KNeighborsClassifier()),
		RandomForestClassifier(n_estimators=10000),
		GradientBoostingClassifier(n_estimators=1000),
		make_pipeline(MinMaxScaler(), MLPClassifier(hidden_layer_sizes=(250, 150))),
	]

	for clf in classifiers:
		clf.fit(x_train, y_train)
		y_pred = clf.predict(x_test)
		acc = accuracy(y_test, y_pred)
		print("Accuracy: {:.2%} \n\n{}\n\n".format(acc, clf))


if __name__ == '__main__':
	train_evaluate()
