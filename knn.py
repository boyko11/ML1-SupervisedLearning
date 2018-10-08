from sklearn import neighbors
from learner import Learner


class KNNLearner(Learner):

    def __init__(self, n_neighbors, weights='uniform', algorithm='auto', n_jobs=1):

        self.estimator = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm, n_jobs=n_jobs)

    def fit_predict_score(self, x_train, y_train, x_test, y_test):

        return super(KNNLearner, self).fit_predict_score(self.estimator, x_train, y_train, x_test, y_test)