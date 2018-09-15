from sklearn import ensemble
from learner import Learner


class BoostingLearner(Learner):

    def __init__(self):
        self.estimator = ensemble.GradientBoostingClassifier()

    def fit_predict_score(self, x_train, y_train, x_test, y_test):

        return super(BoostingLearner, self).fit_predict_score(self.estimator, x_train, y_train, x_test, y_test)