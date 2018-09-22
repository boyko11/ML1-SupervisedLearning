from sklearn import datasets,  model_selection, preprocessing
import numpy as np


def load_and_split_data(scale_data=False, test_size=0.2):

    X, Y = load_data(scale_data=scale_data)
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=test_size)
    return X_train, X_test, Y_train, Y_test


def load_data(scale_data=False):

    data = datasets.load_breast_cancer()
    X = data.data
    Y = data.target
    if scale_data:
        X = preprocessing.scale(X)

    shuffled_indices = np.random.choice(Y.shape[0], Y.shape[0], replace=False)

    X_shuffled = X[shuffled_indices, :]
    Y_shuffled = Y[shuffled_indices]

    print(X_shuffled.shape)
    print(Y_shuffled.shape)

    return X_shuffled, Y_shuffled
