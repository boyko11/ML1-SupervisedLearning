import numpy as np
from data_service import load_and_split_data
from decision_tree import DTLearner
from svm import SVMLearner
from neural_network import NNLearner
from knn import KNNLearner


dt_learner = DTLearner()
svm_learner = SVMLearner()
nn_learner = NNLearner()
knn_learner = KNNLearner(n_neighbors=5, weights='distance')


dt_accuracy_scores = []
svm_accuracy_scores = []
nn_accuracy_scores = []
knn_accuracy_scores = []
num_runs = 10
for i in range(num_runs):
    x_train, x_test, y_train, y_test = load_and_split_data()

    # svm_accuracy_score = svm_learner.fit_predict_score(x_train, y_train, x_test, y_test)
    # svm_accuracy_scores.append(svm_accuracy_score)
    #
    # dt_accuracy_score = dt_learner.fit_predict_score(x_train, y_train, x_test, y_test)
    # dt_accuracy_scores.append(dt_accuracy_score)
    #
    # nn_accuracy_score = nn_learner.fit_predict_score(x_train, y_train, x_test, y_test)
    # nn_accuracy_scores.append(nn_accuracy_score)

    knn_accuracy_score = knn_learner.fit_predict_score(x_train, y_train, x_test, y_test)
    knn_accuracy_scores.append(knn_accuracy_score)

# print("SVM Mean accuracy: {0}".format(np.mean(np.asarray(svm_accuracy_scores))))
# print("DT Mean accuracy: {0}".format(np.mean(np.asarray(dt_accuracy_scores))))
# print("NN Mean accuracy: {0}".format(np.mean(np.asarray(nn_accuracy_scores))))
print("KNN Mean accuracy: {0}".format(np.mean(np.asarray(knn_accuracy_scores))))