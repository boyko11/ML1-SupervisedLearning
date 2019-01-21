import numpy as np
import data_service
from decision_tree import DTLearner
from svm import SVMLearner
from neural_network import NNLearner
from knn import KNNLearner
from boosting import BoostingLearner
from learning_curve_service import plot_learning_curve
from sklearn.multiclass import OneVsRestClassifier
import sys


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


if len(sys.argv) < 3:
    print('Need to specify algorithm and dataset, e.g. python run_learning_curves_compare.py dt breast_cancer')
    exit()

algo = sys.argv[1]
dataset = sys.argv[2]

scale_data = True
transform_data = False
random_slice = None
random_seed=None
title = 'Breast Cancer {0} Learning Curve'.format(algo.upper())

if dataset == 'kdd':
    transform_data = True
    random_slice = 1000
    random_seed = None
    title = 'KDD {0} Learning Curve'.format(algo.upper())


X, Y = data_service.load_data(random_seed=random_seed, dataset=dataset, scale_data=True,
                              transform_data=transform_data, random_slice=random_slice)
X_non_scaled, Y_non_scaled = data_service.load_data(random_seed=random_seed, dataset=dataset, scale_data=False,
                              transform_data=transform_data, random_slice=random_slice)

ylim = (0.0, 0.2)
train_sizes = np.linspace(.1, 1.0, 10)

draw_non_scaled = False
estimator_to_use_non_scaled = None
if algo.upper() == 'DT':
    estimator_to_use = dt_learner.estimator
elif algo.upper() == 'BOOST':
    estimator_to_use = boosting_learner.estimator
elif algo.upper() == 'KNN':
    estimator_to_use = knn_learner.estimator
elif algo.upper() == 'NN':
    estimator_to_use = nn_learner.estimator
    estimator_to_use_non_scaled = nn_learner_non_scaled.estimator
    draw_non_scaled = True
elif algo.upper() == 'SVM':
    estimator_to_use = svm_learner.estimator
    estimator_to_use_non_scaled = svm_learner_non_scaled.estimator
    draw_non_scaled = True

plot_learning_curve(estimator=estimator_to_use, estimator_non_scaled=estimator_to_use_non_scaled,
                    title=title, X=X, y=Y, X_non_scaled=X_non_scaled, Y_non_scaled=Y_non_scaled, ylim=ylim,
                     train_sizes=train_sizes, draw_non_scaled=draw_non_scaled)

