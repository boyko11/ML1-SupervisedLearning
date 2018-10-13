import numpy as np
import data_service
from decision_tree import DTLearner
from svm import SVMLearner
from neural_network import NNLearner
from knn import KNNLearner
from boosting import BoostingLearner
from learning_curve_service import plot_learning_curve
from sklearn.multiclass import OneVsRestClassifier


dt_learner = DTLearner()
dt_learnerOnevsRest = OneVsRestClassifier(dt_learner.estimator)

#--------------------------------
nn_hidden_layer_sizes = (100,)
nn_solver = 'lbfgs'
nn_activation = 'relu'
alpha = 0.0001  # regularization term coefficient
nn_learning_rate = 'constant'
nn_learning_rate_init = 0.0001
nn_learner = NNLearner(hidden_layer_sizes=nn_hidden_layer_sizes, max_iter=200, solver=nn_solver,
                       activation=nn_activation,
                       alpha=alpha, learning_rate=nn_learning_rate, learning_rate_init=nn_learning_rate_init)
nn_learner_non_scaled = NNLearner(hidden_layer_sizes=nn_hidden_layer_sizes, max_iter=200, solver=nn_solver,
                       activation=nn_activation,
                       alpha=alpha, learning_rate=nn_learning_rate, learning_rate_init=nn_learning_rate_init)

#-------------------------------
n_neighbors = 5
weights = 'distance'
algorithm = 'auto'
n_jobs = 5
knn_learner = KNNLearner(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm, n_jobs=n_jobs)

#-------------------------------
class_weight='balanced'
max_depth_boost = 10
n_estimators=50
boosting_learner = BoostingLearner(n_estimators=n_estimators, max_depth=max_depth_boost, class_weight=class_weight)

#------------------------------
kernel = 'linear' #‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’,
C = 1.0
gamma = 'auto'
max_iter = 1000
verbose=False
svm_learner = SVMLearner(kernel=kernel, C=C, max_iter=max_iter)
svm_learner_non_scaled = SVMLearner(kernel=kernel, C=C, max_iter=max_iter)

#-------------------------------
scale_data = False
transform_data = True
random_slice = 10000
random_seed=11
dataset = 'kdd'
title = 'SVM KDD Learning Curve'

X, Y = data_service.load_data(random_seed=random_seed, dataset=dataset, scale_data=True,
                              transform_data=transform_data, random_slice=random_slice)
X_non_scaled, Y_non_scaled = data_service.load_data(random_seed=random_seed, dataset=dataset, scale_data=False,
                              transform_data=transform_data, random_slice=random_slice)

ylim = (0.0, 0.2)
train_sizes = np.linspace(.1, 1.0, 10)

estimator_to_use = svm_learner.estimator
estimator_to_use_non_scaled = svm_learner_non_scaled.estimator

plot_learning_curve(estimator=estimator_to_use, estimator_non_scaled=estimator_to_use_non_scaled,
                    title=title, X=X, y=Y, X_non_scaled=X_non_scaled, Y_non_scaled=Y_non_scaled, ylim=ylim,
                     train_sizes=train_sizes, draw_non_scaled=True)

