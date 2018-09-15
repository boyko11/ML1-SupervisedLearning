from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


def load_and_split_data():

    data = load_breast_cancer()
    X = data.data
    Y = data.target

    X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size=0.2)
    return X_train, X_test, Y_train, Y_test