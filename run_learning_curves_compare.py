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

svm_learner = SVMLearner()

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
algorithm = 'auto'
n_jobs = 5
knn_learner = KNNLearner(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm, n_jobs=n_jobs)

class_weight='balanced'
max_depth_boost = 10
n_estimators=50
boosting_learner = BoostingLearner(n_estimators=n_estimators, max_depth=max_depth_boost, class_weight=class_weight)

scale_data = True
transform_data = True
random_slice = None
random_seed=None
dataset = 'breast_cancer'
title = 'NN Breast Cancer Learning Curve'
X, Y = data_service.load_data(random_seed=random_seed, dataset=dataset, scale_data=scale_data, transform_data=transform_data, random_slice=random_slice)

ylim = (0.0, 0.08)
train_sizes = np.linspace(.1, 1.0, 10)

estimator_to_use = knn_learner.estimator
plot_learning_curve(estimator=estimator_to_use, title=title, X=X, y=Y, ylim=ylim,
                    train_sizes=train_sizes)

