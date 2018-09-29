from sklearn import neighbors
from learner import Learner


class KNNLearner(Learner):

    def __init__(self, n_neighbors, weights='distance'):

        self.n_neighbors = n_neighbors
        self.weights = weights
        self.estimator = neighbors.KNeighborsClassifier(n_neighbors=self.n_neighbors, weights=self.weights)

    def fit_predict_score(self, x_train, y_train, x_test, y_test):

        return super(KNNLearner, self).fit_predict_score(self.estimator, x_train, y_train, x_test, y_test)