import numpy as np
import data_service
from svm import SVMLearner
from neural_network import NNLearner
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve


def plot_iter_learning_curve(title, num_iter_list, train_accuracy_scores, test_accuracy_scores, ylim=None):

    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Iterations")
    plt.ylabel("Error")

    num_iter_nparray = np.asarray(num_iter_list)
    train_accuracy_scores_nparray = np.asarray(train_accuracy_scores)
    test_accuracy_scores_nparray = np.asarray(test_accuracy_scores)

    train_errors = 1 - train_accuracy_scores_nparray
    test_errors = 1 - test_accuracy_scores_nparray

    plt.grid()

    plt.plot(num_iter_nparray, train_errors, color="r", label="Training error")
    plt.plot(num_iter_nparray, test_errors, color="g", label="Test error")

    plt.legend(loc="best")
    plt.show()


max_number_iter = 101
iter_step = 1
ylim = (0.0, 0.3)

#try scaling vs not scaling, different train/test splits?
X, Y = data_service.load_data(scale_data=False)
num_iter_list = []
nn_train_accuracy_scores, svm_train_accuracy_scores = [], []
nn_test_accuracy_scores, svm_test_accuracy_scores = [], []

for i in range(1, max_number_iter, iter_step):

    print(i)
    svm_learner = SVMLearner(max_iter=i)
    nn_learner = NNLearner(max_iter=i)

    train_sizes, nn_train_scores, nn_test_scores = learning_curve(
        nn_learner.estimator, X, Y, train_sizes=np.array([.8]))

    train_sizes, svm_train_scores, svm_test_scores = learning_curve(
        svm_learner.estimator, X, Y, train_sizes=np.array([.8]))

    num_iter_list.append(i)
    nn_test_accuracy_scores.append(np.mean(nn_test_scores))
    nn_train_accuracy_scores.append(np.mean(nn_train_scores))
    svm_test_accuracy_scores.append(np.mean(svm_test_scores))
    svm_train_accuracy_scores.append(np.mean(svm_train_scores))


plot_iter_learning_curve('NN Iterations Learning Curve', num_iter_list, nn_train_accuracy_scores,
                         nn_test_accuracy_scores, ylim)
plot_iter_learning_curve('SVM Iterations Learning Curve', num_iter_list, svm_train_accuracy_scores,
                         svm_test_accuracy_scores, ylim)
