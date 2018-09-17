import numpy as np
import data_service
import stats_service
from decision_tree import DTLearner
from svm import SVMLearner
from neural_network import NNLearner
from knn import KNNLearner
from boosting import BoostingLearner


dt_learner = DTLearner()
svm_learner = SVMLearner()
nn_learner = NNLearner()
knn_learner = KNNLearner(n_neighbors=5, weights='distance')
boosting_learner = BoostingLearner()


dt_accuracy_scores, svm_accuracy_scores, nn_accuracy_scores, knn_accuracy_scores, boosting_accuracy_scores = [], [], [], [], []
dt_fit_times, svm_fit_times, nn_fit_times, knn_fit_times, boosting_fit_times = [], [], [], [], []
dt_predict_times, svm_predict_times, nn_predict_times, knn_predict_times, boosting_predict_times = [], [], [], [], []

num_runs = 10
scale_data = False

for i in range(num_runs):
    x_train, x_test, y_train, y_test = data_service.load_and_split_data(scale_data=scale_data)

    svm_accuracy_score, svm_fit_time, svm_predict_time = svm_learner.fit_predict_score(x_train, y_train, x_test, y_test)
    stats_service.record_stats(svm_accuracy_scores, svm_accuracy_score, svm_fit_times, svm_fit_time, svm_predict_times,
                               svm_predict_time)

    dt_accuracy_score, dt_fit_time, dt_predict_time = dt_learner.fit_predict_score(x_train, y_train, x_test, y_test)
    stats_service.record_stats(dt_accuracy_scores, dt_accuracy_score, dt_fit_times, dt_fit_time, dt_predict_times,
                               dt_predict_time)

    nn_accuracy_score, nn_fit_time, nn_predict_time = nn_learner.fit_predict_score(x_train, y_train, x_test, y_test)
    stats_service.record_stats(nn_accuracy_scores, nn_accuracy_score, nn_fit_times, nn_fit_time, nn_predict_times,
                               nn_predict_time)

    knn_accuracy_score, knn_fit_time, knn_predict_time = knn_learner.fit_predict_score(x_train, y_train, x_test, y_test)
    stats_service.record_stats(knn_accuracy_scores, knn_accuracy_score, knn_fit_times, knn_fit_time, knn_predict_times,
                               knn_predict_time)

    boosting_accuracy_score, boosting_fit_time, boosting_predict_time = boosting_learner.fit_predict_score(x_train,
                                                                                           y_train, x_test, y_test)
    stats_service.record_stats(boosting_accuracy_scores, boosting_accuracy_score, boosting_fit_times, boosting_fit_time,
                               boosting_predict_times, boosting_predict_time)


print("SVM Mean Accuracy: {0}".format(stats_service.mean(svm_accuracy_scores)))
print("DT Mean Accuracy: {0}".format(stats_service.mean(dt_accuracy_scores)))
print("NN Mean Accuracy: {0}".format(stats_service.mean(nn_accuracy_scores)))
print("KNN Mean accuracy: {0}".format(stats_service.mean(knn_accuracy_scores)))
print("Boosting Mean accuracy: {0}".format(stats_service.mean(boosting_accuracy_scores)))

print('----------------------------------------------------')

print("SVM Mean Fit Time: {0}".format(stats_service.mean(svm_fit_times)))
print("DT Mean Fit Time: {0}".format(stats_service.mean(dt_fit_times)))
print("NN Mean Fit Time: {0}".format(stats_service.mean(nn_fit_times)))
print("KNN Mean Fit Time: {0}".format(stats_service.mean(knn_fit_times)))
print("Boosting Mean Fit Time: {0}".format(stats_service.mean(boosting_fit_times)))

print('----------------------------------------------------')

print("SVM Mean Predict Time: {0}".format(stats_service.mean(svm_predict_times)))
print("DT Mean Predict Time: {0}".format(stats_service.mean(dt_predict_times)))
print("NN Mean Predict Time: {0}".format(stats_service.mean(nn_predict_times)))
print("KNN Mean Predict Time: {0}".format(stats_service.mean(knn_predict_times)))
print("Boosting Mean Predict Time: {0}".format(stats_service.mean(boosting_predict_times)))
