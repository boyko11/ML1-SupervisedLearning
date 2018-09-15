from sklearn import tree
from learner import Learner


class DTLearner(Learner):

    def __init__(self):
        self.estimator = tree.DecisionTreeClassifier(random_state=78)

    def fit_predict_score(self, x_train, y_train, x_test, y_test):

        return super(DTLearner, self).fit_predict_score(self.estimator, x_train, y_train, x_test, y_test)




