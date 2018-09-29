from sklearn import svm
from learner import Learner


class SVMLearner(Learner):

    def __init__(self, max_iter=1000, kernel='rbf', gamma='auto', C=1.0):
        self.estimator = svm.LinearSVC(random_state=2, max_iter=max_iter)
        #self.estimator = svm.SVC(kernel=kernel, gamma=gamma, C=C)

    def fit_predict_score(self, x_train, y_train, x_test, y_test):

        return super(SVMLearner, self).fit_predict_score(self.estimator, x_train, y_train, x_test, y_test)
