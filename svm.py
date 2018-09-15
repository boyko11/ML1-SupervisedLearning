from sklearn import svm
from learner import Learner


class SVMLearner(Learner):

    def __init__(self):
        self.estimator = svm.LinearSVC(random_state=2)

    def fit_predict_score(self, x_train, y_train, x_test, y_test):

        return super(SVMLearner, self).fit_predict_score(self.estimator, x_train, y_train, x_test, y_test)
