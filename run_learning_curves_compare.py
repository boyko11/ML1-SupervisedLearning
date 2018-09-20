import numpy as np
import data_service
from decision_tree import DTLearner
from svm import SVMLearner
from neural_network import NNLearner
from knn import KNNLearner
from boosting import BoostingLearner
from learning_curve_service import plot_learning_curve


dt_learner = DTLearner()
svm_learner = SVMLearner()
nn_learner = NNLearner()
knn_learner = KNNLearner(n_neighbors=5, weights='distance')
boosting_learner = BoostingLearner()

X, Y = data_service.load_data()

shuffled_indices = np.random.choice(Y.shape[0], Y.shape[0], replace=False)

X_shuffled = X[shuffled_indices, :]
Y_shuffled = Y[shuffled_indices]

print(X_shuffled.shape)
print(Y_shuffled.shape)

ylim = (0.0, 0.65)
train_sizes = np.linspace(.1, 1.0, 10)

plot_learning_curve(estimator=dt_learner.estimator, title='DT Learning Curve', X=X_shuffled, y=Y_shuffled, ylim=ylim,
                    train_sizes=train_sizes)

plot_learning_curve(estimator=svm_learner.estimator, title='SVM Learning Curve', X=X_shuffled, y=Y_shuffled, ylim=ylim,
                    train_sizes=train_sizes)

plot_learning_curve(estimator=nn_learner.estimator, title='NN Learning Curve', X=X_shuffled, y=Y_shuffled, ylim=ylim,
                    train_sizes=train_sizes)

plot_learning_curve(estimator=knn_learner.estimator, title='KNN Learning Curve', X=X_shuffled, y=Y_shuffled, ylim=ylim,
                    train_sizes=train_sizes)

plot_learning_curve(estimator=boosting_learner.estimator, title='Boosting Learning Curve', X=X_shuffled, y=Y_shuffled,
                    ylim=ylim, train_sizes=train_sizes)
