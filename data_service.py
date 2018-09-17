from sklearn import datasets,  model_selection, preprocessing


def load_and_split_data(scale_data=False):

    X, Y = load_data(scale_data=scale_data)
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split( X, Y, test_size=0.2)
    return X_train, X_test, Y_train, Y_test


def load_data(scale_data=False):

    data = datasets.load_breast_cancer()
    X = data.data
    Y = data.target
    if scale_data:
        X = preprocessing.scale(X)
    return X, Y
