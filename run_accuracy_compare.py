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
from sklearn.multiclass import OneVsRestClassifier

dt_accuracy_scores, svm_accuracy_scores, nn_accuracy_scores, knn_accuracy_scores, boosting_accuracy_scores = [], [], [], [], []
dt_fit_times, svm_fit_times, nn_fit_times, knn_fit_times, boosting_fit_times = [], [], [], [], []
dt_predict_times, svm_predict_times, nn_predict_times, knn_predict_times, boosting_predict_times = [], [], [], [], []

##  DT #############################
criterion='gini'
min_samples_leaf=1
max_depth=None
class_weight='balanced'
dt_learner = DTLearner(criterion=criterion, min_samples_leaf=min_samples_leaf, max_depth=max_depth, class_weight=class_weight)
# dt_learnerOnevsRest = OneVsRestClassifier(dt_learner.estimator)
# dt_learner.estimator = dt_learnerOnevsRest


##  Boost #############################
class_weight='balanced'
max_depth_boost = 10
n_estimators=50
boosting_learner = BoostingLearner(n_estimators=n_estimators, max_depth=max_depth_boost, class_weight=class_weight)

##  KNN #############################
n_neighbors = 5
weights = 'distance'
algorithm = 'auto'
n_jobs = 1
knn_learner = KNNLearner(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm, n_jobs=n_jobs)

##  NN #############################
# nn_hidden_layer_sizes = (100,)
# nn_solver = 'lbfgs'
# nn_activation = 'relu'
# alpha = 0.0001 #regularization term coefficient
# nn_learning_rate = 'constant'
# nn_learning_rate_init = 0.0001
nn_activation = 'relu'
alpha = 0.0001
nn_hidden_layer_sizes = (100,)
nn_learning_rate = 'constant'
nn_learning_rate_init = 0.01
nn_solver = 'lbfgs'

#{'activation': 'relu', 'alpha': 0.0001, 'hidden_layer_sizes': (100,), 'learning_rate_init': 0.01, 'solver': 'lbfgs'}

nn_learner = NNLearner(hidden_layer_sizes=nn_hidden_layer_sizes, max_iter=200, solver=nn_solver, activation=nn_activation,
                       alpha=alpha, learning_rate=nn_learning_rate, learning_rate_init=nn_learning_rate_init)

##  SVM #############################
# kernel = 'linear' #‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’,
# C = 1.0
# gamma = 'auto'
# max_iter = 200
# verbose=False
# poly_degree=3
# cache_size=2000
# class_weight='balanced'
#{'C': 10, 'class_weight': 'balanced', 'gamma': 0.001, 'kernel': 'rbf'}
kernel = 'rbf' #‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’,
C = 10
gamma = 0.001
max_iter = 200
verbose=False
poly_degree=3
cache_size=200
class_weight='balanced'
svm_learner = SVMLearner(kernel=kernel, C=C, max_iter=max_iter, degree=poly_degree, cache_size=cache_size, class_weight=class_weight)
#svm_learner = SVMLearner(kernel=kernel, C=C, gamma=gamma, max_iter=max_iter, verbose=verbose)

num_runs = 1
scale_data = True
transform_data = True
random_slice = None
random_seed = 777

dataset = 'kdd'
test_size = 0.5

for i in range(num_runs):
    x_train, x_test, y_train, y_test = data_service.load_and_split_data(scale_data=scale_data, transform_data=transform_data,
                                                                        random_slice=random_slice, random_seed=random_seed, dataset=dataset, test_size=test_size)

    x_train_copy1, x_train_copy2, x_train_copy3, x_train_copy4 = x_train.copy(), x_train.copy(), x_train.copy(), x_train.copy()
    y_train_copy1, y_train_copy2, y_train_copy3, y_train_copy4 = y_train.copy(), y_train.copy(), y_train.copy(), y_train.copy()
    x_test_copy1, x_test_copy2, x_test_copy3, x_test_copy4 = x_test.copy(), x_test.copy(), x_test.copy(), x_test.copy()
    y_test_copy1, y_test_copy2, y_test_copy3, y_test_copy4 = y_test.copy(), y_test.copy(), y_test.copy(), y_test.copy()

    # dt_accuracy_score, dt_fit_time, dt_predict_time = dt_learner.fit_predict_score(x_train, y_train, x_test, y_test)
    # print('dt {3}: {0}, {1}, {2}'.format(dt_accuracy_score, dt_fit_time, dt_predict_time, i))
    # stats_service.record_stats(dt_accuracy_scores, dt_accuracy_score, dt_fit_times, dt_fit_time, dt_predict_times,
    #                            dt_predict_time)
    # dt_learner.draw_tree(tree_id=i)
    #
    # knn_accuracy_score, knn_fit_time, knn_predict_time = knn_learner.fit_predict_score(x_train_copy1, y_train_copy1, x_test_copy1, y_test_copy1)
    # print('knn: {0}, {1}, {2}'.format(knn_accuracy_score, knn_fit_time, knn_predict_time))
    # stats_service.record_stats(knn_accuracy_scores, knn_accuracy_score, knn_fit_times, knn_fit_time, knn_predict_times,
    #                            knn_predict_time)
    #
    #
    # boosting_accuracy_score, boosting_fit_time, boosting_predict_time = boosting_learner.fit_predict_score(x_train_copy2, y_train_copy2, x_test_copy2, y_test_copy2)
    # print('boost {3}: {0}, {1}, {2}'.format(boosting_accuracy_score, boosting_fit_time, boosting_predict_time, i))
    # stats_service.record_stats(boosting_accuracy_scores, boosting_accuracy_score, boosting_fit_times, boosting_fit_time,
    #                            boosting_predict_times, boosting_predict_time)
    #
    nn_accuracy_score, nn_fit_time, nn_predict_time = nn_learner.fit_predict_score(x_train_copy3, y_train_copy3, x_test_copy3, y_test_copy3)
    print('nn: {0}, {1}, {2}'.format(nn_accuracy_score, nn_fit_time, nn_predict_time))
    stats_service.record_stats(nn_accuracy_scores, nn_accuracy_score, nn_fit_times, nn_fit_time, nn_predict_times,
                               nn_predict_time)

    # svm_accuracy_score, svm_fit_time, svm_predict_time = svm_learner.fit_predict_score(x_train_copy4, y_train_copy4, x_test_copy4, y_test_copy4)
    # print('svm: {0}, {1}, {2}'.format(svm_accuracy_score, svm_fit_time, svm_predict_time))
    # stats_service.record_stats(svm_accuracy_scores, svm_accuracy_score, svm_fit_times, svm_fit_time, svm_predict_times,
    #                            svm_predict_time)



print('-----------------------------------------------')
print("SVM: {0}, {1}, {2}".format(stats_service.mean(svm_accuracy_scores), stats_service.mean(svm_fit_times), stats_service.mean(svm_predict_times)))
print("DT: {0}, {1}, {2}".format(stats_service.mean(dt_accuracy_scores), stats_service.mean(dt_fit_times), stats_service.mean(dt_predict_times)))
print("NN: {0}, {1}, {2}".format(stats_service.mean(nn_accuracy_scores), stats_service.mean(nn_fit_times), stats_service.mean(nn_predict_times)))
print("KNN: {0}, {1}, {2}".format(stats_service.mean(knn_accuracy_scores), stats_service.mean(knn_fit_times), stats_service.mean(knn_predict_times)))
print("Boosting: {0}, {1}, {2}".format(stats_service.mean(boosting_accuracy_scores), stats_service.mean(boosting_fit_times), stats_service.mean(boosting_predict_times)))

print('----------------------------------------------------')

#boosting_learner.display_trees()
