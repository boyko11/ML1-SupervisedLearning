import numpy as np
import data_service
from svm import SVMLearner
from neural_network import NNLearner
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve


def plot_iter_learning_curve(title, num_iter_list, train_accuracy_scores, test_accuracy_scores,
                             train_accuracy_scores_scaled, test_accuracy_scores_scaled, ylim=None):

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

    train_accuracy_scores_scaled_nparray = np.asarray(train_accuracy_scores_scaled)
    test_accuracy_scores_scaled_nparray = np.asarray(test_accuracy_scores_scaled)

    train_errors_scaled = 1 - train_accuracy_scores_scaled_nparray
    test_errors_scaled = 1 - test_accuracy_scores_scaled_nparray

    plt.grid()

    plt.plot(num_iter_nparray, train_errors_scaled, color="r", label="Training error Scaled", linestyle='solid')
    plt.plot(num_iter_nparray, test_errors_scaled, color="g", label="Test error Scaled", linestyle='solid')

    plt.plot(num_iter_nparray, train_errors, color="r", label="Training error Non-Scaled", linestyle='dashed')
    plt.plot(num_iter_nparray, test_errors, color="g", label="Test error Non-Scaled", linestyle='dashed')

    plt.legend(loc="best")
    plt.show()


max_number_iter = 311
iter_step = 5
ylim = (0.0, 0.3)
random_seed = 777
dataset = 'kdd'
transform_data = True
random_slice = 10000
test_size = 0.5

#try scaling vs not scaling, different train/test splits?
X, Y = data_service.load_data(scale_data=False, random_seed=random_seed, transform_data=transform_data,
                              dataset=dataset, random_slice=random_slice)
X_scaled, Y_scaled = data_service.load_data(scale_data=True, random_seed=random_seed,transform_data=transform_data,
                                            dataset=dataset, random_slice=random_slice)
num_iter_list = []
nn_train_accuracy_scores, nn_train_accuracy_scores_scaled, svm_train_accuracy_scores = [], [], []
nn_test_accuracy_scores, nn_test_accuracy_scores_scaled, svm_test_accuracy_scores = [], [], []

for i in range(1, max_number_iter, iter_step):

    print(i)
    # svm_learner = SVMLearner(max_iter=i)
    nn_learner = NNLearner(max_iter=i)

    nn_hidden_layer_sizes = (100,)
    nn_solver = 'lbfgs'
    nn_activation = 'relu'
    alpha = 0.0001  # regularization term coefficient
    nn_learning_rate = 'constant'
    nn_learning_rate_init = 0.0001
    nn_learner = NNLearner(hidden_layer_sizes=nn_hidden_layer_sizes, max_iter=i, solver=nn_solver,
                           activation=nn_activation,
                           alpha=alpha, learning_rate=nn_learning_rate, learning_rate_init=nn_learning_rate_init)

    train_sizes, nn_train_scores, nn_test_scores = learning_curve(
        nn_learner.estimator, X, Y, train_sizes=np.array([1 - test_size]))

    nn_learner_scaled = NNLearner(hidden_layer_sizes=nn_hidden_layer_sizes, max_iter=i, solver=nn_solver,
                           activation=nn_activation,
                           alpha=alpha, learning_rate=nn_learning_rate, learning_rate_init=nn_learning_rate_init)

    train_sizes_scaled, nn_train_scores_scaled, nn_test_scores_scaled = learning_curve(
        nn_learner_scaled.estimator, X_scaled, Y_scaled, train_sizes=np.array([1 - test_size]))

    # train_sizes, svm_train_scores, svm_test_scores = learning_curve(
    #     svm_learner.estimator, X, Y, train_sizes=np.array([.9]))

    num_iter_list.append(i)
    nn_test_accuracy_scores.append(np.mean(nn_test_scores))
    nn_train_accuracy_scores.append(np.mean(nn_train_scores))
    nn_test_accuracy_scores_scaled.append(np.mean(nn_test_scores_scaled))
    nn_train_accuracy_scores_scaled.append(np.mean(nn_train_scores_scaled))
    # svm_test_accuracy_scores.append(np.mean(svm_test_scores))
    # svm_train_accuracy_scores.append(np.mean(svm_train_scores))


plot_iter_learning_curve('NN Iterations Learning Curve', num_iter_list, nn_train_accuracy_scores,
                         nn_test_accuracy_scores, nn_train_accuracy_scores_scaled, nn_test_accuracy_scores_scaled, ylim)
# plot_iter_learning_curve('SVM Iterations Learning Curve', num_iter_list, svm_train_accuracy_scores,
#                          svm_test_accuracy_scores, ylim)
