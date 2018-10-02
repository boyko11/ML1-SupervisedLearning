# dt: 0.9434524065643745
# knn: 0.9331428620603599
# boost: 0.5588926275569478
# nn: 0.8255208557438276



import numpy as np
import data_service
import stats_service
from decision_tree import DTLearner
from svm import SVMLearner
from neural_network import NNLearner
from knn import KNNLearner
from boosting import BoostingLearner


dt_learner = DTLearner(criterion='entropy', min_samples_leaf=25, max_depth=14)
kernel = 'linear' #‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’,
C = 1.0
gamma = 'auto'
max_iter = 1000
verbose=False
svm_learner = SVMLearner(kernel=kernel, C=C)
#svm_learner = SVMLearner(kernel=kernel, C=C, gamma=gamma, max_iter=max_iter, verbose=verbose)
nn_hidden_layer_sizes = (100,)
nn_solver = 'lbfgs'
nn_activation = 'relu'
alpha = 0.0001 #regularization term coefficient
nn_learning_rate = 'constant'
nn_learning_rate_init = 0.0001
nn_learner = NNLearner(hidden_layer_sizes=nn_hidden_layer_sizes, max_iter=200, solver=nn_solver, activation=nn_activation,
                       alpha=alpha, learning_rate=nn_learning_rate, learning_rate_init=nn_learning_rate_init)
n_neighbors = 5
weights = 'distance'
knn_learner = KNNLearner(n_neighbors=n_neighbors, weights=weights)
boosting_learner = BoostingLearner()


dt_accuracy_scores, svm_accuracy_scores, nn_accuracy_scores, knn_accuracy_scores, boosting_accuracy_scores = [], [], [], [], []
dt_fit_times, svm_fit_times, nn_fit_times, knn_fit_times, boosting_fit_times = [], [], [], [], []
dt_predict_times, svm_predict_times, nn_predict_times, knn_predict_times, boosting_predict_times = [], [], [], [], []

num_runs = 1
scale_data = True
# random_slice = 100000
# random_seed = 77
random_slice = 100000
random_seed = 17
dataset = 'kdd'

for i in range(num_runs):
    x_train, x_test, y_train, y_test = data_service.load_and_split_data(scale_data=scale_data,
                                                                        random_slice=random_slice, random_seed=random_seed, dataset=dataset)

    dt_accuracy_score, dt_fit_time, dt_predict_time = dt_learner.fit_predict_score(x_train, y_train, x_test, y_test)
    print('dt: {0}, {1}, {2}'.format(dt_accuracy_score, dt_fit_time, dt_predict_time))
    stats_service.record_stats(dt_accuracy_scores, dt_accuracy_score, dt_fit_times, dt_fit_time, dt_predict_times,
                               dt_predict_time)
    dt_learner.draw_tree()

    # knn_accuracy_score, knn_fit_time, knn_predict_time = knn_learner.fit_predict_score(x_train, y_train, x_test, y_test)
    # print('knn: {0}, {1}, {2}'.format(knn_accuracy_score, knn_fit_time, knn_predict_time))
    # stats_service.record_stats(knn_accuracy_scores, knn_accuracy_score, knn_fit_times, knn_fit_time, knn_predict_times,
    #                            knn_predict_time)
    #
    # boosting_accuracy_score, boosting_fit_time, boosting_predict_time = boosting_learner.fit_predict_score(x_train,
    #                                                                                        y_train, x_test, y_test)
    # print('boost: {0}, {1}, {2}'.format(boosting_accuracy_score, boosting_fit_time, boosting_predict_time))
    # stats_service.record_stats(boosting_accuracy_scores, boosting_accuracy_score, boosting_fit_times, boosting_fit_time,
    #                            boosting_predict_times, boosting_predict_time)
    #
    # nn_accuracy_score, nn_fit_time, nn_predict_time = nn_learner.fit_predict_score(x_train, y_train, x_test, y_test)
    # print('nn: {0}, {1}, {2}'.format(nn_accuracy_score, nn_fit_time, nn_predict_time))
    # stats_service.record_stats(nn_accuracy_scores, nn_accuracy_score, nn_fit_times, nn_fit_time, nn_predict_times,
    #                            nn_predict_time)
    #
    # svm_accuracy_score, svm_fit_time, svm_predict_time = svm_learner.fit_predict_score(x_train, y_train, x_test, y_test)
    # print('svm: {0}, {1}, {2}'.format(svm_accuracy_score, svm_fit_time, svm_predict_time))
    # stats_service.record_stats(svm_accuracy_scores, svm_accuracy_score, svm_fit_times, svm_fit_time, svm_predict_times,
    #                            svm_predict_time)




print("SVM: {0}, {1}, {2}".format(stats_service.mean(svm_accuracy_scores), stats_service.mean(svm_fit_times), stats_service.mean(svm_predict_times)))
print("DT: {0}, {1}, {2}".format(stats_service.mean(dt_accuracy_scores), stats_service.mean(dt_fit_times), stats_service.mean(dt_predict_times)))
print("NN: {0}, {1}, {2}".format(stats_service.mean(nn_accuracy_scores), stats_service.mean(nn_fit_times), stats_service.mean(nn_predict_times)))
print("KNN: {0}, {1}, {2}".format(stats_service.mean(knn_accuracy_scores), stats_service.mean(knn_fit_times), stats_service.mean(knn_predict_times)))
print("Boosting: {0}, {1}, {2}".format(stats_service.mean(boosting_accuracy_scores), stats_service.mean(boosting_fit_times), stats_service.mean(boosting_predict_times)))

print('----------------------------------------------------')

#boosting_learner.display_trees()
