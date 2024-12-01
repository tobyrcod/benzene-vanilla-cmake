###
### Demos from: https://github.com/cair/pyTsetlinMachine
###

from pyTsetlinMachine.tm import MultiClassTsetlinMachine
from pyTsetlinMachine.tools import Binarizer
import numpy as np

from sklearn import datasets
from sklearn.model_selection import train_test_split
import tensorflow as tf
import time

def interpretability_xor_demo():
	number_of_features = 20
	noise = 0.1

	X_train = np.random.randint(0, 2, size=(5000, number_of_features), dtype=np.uint32)
	Y_train = np.logical_xor(X_train[:,0], X_train[:,1]).astype(dtype=np.uint32)
	Y_train = np.where(np.random.rand(5000) <= noise, 1-Y_train, Y_train) # Adds noise

	X_test = np.random.randint(0, 2, size=(5000, number_of_features), dtype=np.uint32)
	Y_test = np.logical_xor(X_test[:,0], X_test[:,1]).astype(dtype=np.uint32)

	tm = MultiClassTsetlinMachine(10, 15, 3.0, boost_true_positive_feedback=0)

	tm.fit(X_train, Y_train, epochs=200)

	print("Accuracy:", 100*(tm.predict(X_test) == Y_test).mean())

	print("\nClass 0 Positive Clauses:\n")
	for j in range(0, 10, 2):
		print("Clause #%d: " % (j), end=' ')
		l = []
		for k in range(number_of_features*2):
			if tm.ta_action(0, j, k) == 1:
				if k < number_of_features:
					l.append(" x%d" % (k))
				else:
					l.append("¬x%d" % (k-number_of_features))
		print(" ∧ ".join(l))

	print("\nClass 0 Negative Clauses:\n")
	for j in range(1, 10, 2):
		print("Clause #%d: " % (j), end=' ')
		l = []
		for k in range(number_of_features*2):
			if tm.ta_action(0, j, k) == 1:
				if k < number_of_features:
					l.append(" x%d" % (k))
				else:
					l.append("¬x%d" % (k-number_of_features))
		print(" ∧ ".join(l))

	print("\nClass 1 Positive Clauses:\n")
	for j in range(0, 10, 2):
		print("Clause #%d: " % (j), end=' ')
		l = []
		for k in range(number_of_features*2):
			if tm.ta_action(1, j, k) == 1:
				if k < number_of_features:
					l.append(" x%d" % (k))
				else:
					l.append("¬x%d" % (k-number_of_features))
		print(" ∧ ".join(l))

	print("\nClass 1 Negative Clauses:\n")
	for j in range(1, 10, 2):
		print("Clause #%d: " % (j), end=' ')
		l = []
		for k in range(number_of_features*2):
			if tm.ta_action(1, j, k) == 1:
				if k < number_of_features:
					l.append(" x%d" % (k))
				else:
					l.append("¬x%d" % (k-number_of_features))
		print(" ∧ ".join(l))

def continuous_breast_cancer_demo():
	breast_cancer = datasets.load_breast_cancer()
	X = breast_cancer.data
	Y = breast_cancer.target

	b = Binarizer(max_bits_per_feature=10)
	b.fit(X)
	X_transformed = b.transform(X)

	tm = MultiClassTsetlinMachine(800, 40, 5.0)

	print("\nMean accuracy over 100 runs:\n")
	tm_results = np.empty(0)
	for i in range(100):
		X_train, X_test, Y_train, Y_test = train_test_split(X_transformed, Y, test_size=0.2)

		tm.fit(X_train, Y_train, epochs=25)
		tm_results = np.append(tm_results, np.array(100 * (tm.predict(X_test) == Y_test).mean()))
		print("#%d Average Accuracy: %.2f%% +/- %.2f" % (
		i + 1, tm_results.mean(), 1.96 * tm_results.std() / np.sqrt(i + 1)))

def mnist_demo():
	(X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.mnist.load_data()

	X_train = np.where(X_train.reshape((X_train.shape[0], 28 * 28)) > 75, 1, 0)
	X_test = np.where(X_test.reshape((X_test.shape[0], 28 * 28)) > 75, 1, 0)

	tm = MultiClassTsetlinMachine(2000, 50, 10.0)

	print("\nAccuracy over 250 epochs:\n")
	for i in range(250):
		start_training = time.time()
		tm.fit(X_train, Y_train, epochs=1, incremental=True)
		stop_training = time.time()

		start_testing = time.time()
		result = 100 * (tm.predict(X_test) == Y_test).mean()
		stop_testing = time.time()

		print("#%d Accuracy: %.2f%% Training: %.2fs Testing: %.2fs" % (
		i + 1, result, stop_training - start_training, stop_testing - start_testing))

interpretability_xor_demo()
