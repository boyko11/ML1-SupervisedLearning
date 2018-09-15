from sklearn import neural_network
from learner import Learner


class NNLearner(Learner):

    def fit_predict_score(self, x_train, y_train, x_test, y_test):

        # mlp_classifier = neural_network.MLPClassifier(hidden_layer_sizes=(64,64))
        mlp_classifier = neural_network.MLPClassifier(solver='lbfgs', alpha=1e-5, random_state=1)

        return super(NNLearner, self).fit_predict_score(mlp_classifier, x_train, y_train, x_test, y_test)



