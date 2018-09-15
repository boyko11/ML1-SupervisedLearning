from sklearn import tree
from learner import Learner


class DTLearner(Learner):

    def fit_predict_score(self, x_train, y_train, x_test, y_test):

        dt_classifier = tree.DecisionTreeClassifier(random_state=78)
        return super(DTLearner, self).fit_predict_score(dt_classifier, x_train, y_train, x_test, y_test)




