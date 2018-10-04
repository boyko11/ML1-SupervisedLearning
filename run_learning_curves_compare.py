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

nn_learner = NNLearner()

knn_learner = KNNLearner(n_neighbors=5, weights='distance')

boosting_learner = BoostingLearner()

scale_data = False
random_slice = None
random_seed=None
title = 'Breast Cancer Boosting Learning Curve'
X, Y = data_service.load_data(random_seed=random_seed, dataset='kdd', scale_data=scale_data, random_slice=random_slice)

ylim = (0.0, 0.65)
ylim = (0.0, 0.03)
train_sizes = np.linspace(.1, 1.0, 10)

plot_learning_curve(estimator=dt_learnerOnevsRest, title=title, X=X, y=Y, ylim=ylim,
                    train_sizes=train_sizes)

# plot_learning_curve(estimator=svm_learner.estimator, title='SVM Learning Curve', X=X, y=Y, ylim=ylim,
#                     train_sizes=train_sizes)
#
# plot_learning_curve(estimator=nn_learner.estimator, title='NN Learning Curve', X=X, y=Y, ylim=ylim,
#                     train_sizes=train_sizes)
#
# plot_learning_curve(estimator=knn_learner.estimator, title='KNN Learning Curve', X=X, y=Y, ylim=ylim,
#                     train_sizes=train_sizes)
#
# plot_learning_curve(estimator=boosting_learner.estimator, title='Boosting Learning Curve', X=X, y=Y,
#                     ylim=ylim, train_sizes=train_sizes)
