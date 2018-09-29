from sklearn import datasets,  model_selection, preprocessing
import numpy as np


def load_and_split_data(scale_data=False, test_size=0.2, random_slice=None, random_seed=None):

    X, Y = load_data(scale_data=scale_data, random_slice=random_slice, random_seed=random_seed)
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=test_size)
    return X_train, X_test, Y_train, Y_test


def load_data(scale_data=False, random_slice=None, random_seed=None):

    if random_seed is not None:
        np.random.seed(random_seed)

    #data = datasets.load_breast_cancer()
    #data = datasets.fetch_covtype()
    data = datasets.fetch_kddcup99()
    X = data.data

    print(X[1:5, :])
    print('-----------------------------------------------')
    Y = data.target
    print(Y[1:5])
    print('-----------------------------------------------')
    print(X.shape)
    print(Y.shape)
    if random_slice is not None:
        random_indices = np.random.choice(Y.shape[0], random_slice, replace=False)
        X = X[random_indices, :]
        Y = Y[random_indices]

    if scale_data:

        for i in range(X.shape[1]):
            #print(i, X[:, i].dtype, isinstance(X[:, i].dtype, float), isinstance(X[:, i].dtype, int), isinstance(X[:, i].dtype, float) or isinstance(X[:, i].dtype, int))
            le = preprocessing.LabelEncoder()
            le.fit(X[:, i])
            X[:, i] = le.transform(X[:, i])

        X = preprocessing.scale(X)

        le = preprocessing.LabelEncoder()
        le.fit(Y)
        Y = le.transform(Y)
        print('Transformed and Scaled:')
        print(X[1:5, :])
        print('------------------------------------------')
        print(Y[1:5])
        print('-----------------------------------------------')

    shuffled_indices = np.random.choice(Y.shape[0], Y.shape[0], replace=False)

    X_shuffled = X[shuffled_indices, :]
    Y_shuffled = Y[shuffled_indices]

    print(X_shuffled.shape)
    print(Y_shuffled.shape)

    return X_shuffled, Y_shuffled
