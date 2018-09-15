from sklearn import neighbors
from learner import Learner


class KNNLearner(Learner):

    def __init__(self, n_neighbors, weights='distance'):
        self.n_neighbors = n_neighbors
        self.weights = weights


    def fit_predict_score(self, x_train, y_train, x_test, y_test):

        # mlp_classifier = neural_network.MLPClassifier(hidden_layer_sizes=(64,64))
        knn_classifier = neighbors.KNeighborsClassifier(self.n_neighbors, self.weights)

        return super(KNNLearner, self).fit_predict_score(knn_classifier, x_train, y_train, x_test, y_test)







#neighbors.KNeighborsClassifier()