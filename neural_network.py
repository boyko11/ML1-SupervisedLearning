from sklearn import neural_network
from learner import Learner


class NNLearner(Learner):

    def __init__(self):
        self.estimator = neural_network.MLPClassifier(solver='lbfgs', alpha=1e-5, random_state=1)

    def fit_predict_score(self, x_train, y_train, x_test, y_test):

        # mlp_classifier = neural_network.MLPClassifier(hidden_layer_sizes=(64,64))
        return super(NNLearner, self).fit_predict_score(self.estimator, x_train, y_train, x_test, y_test)



