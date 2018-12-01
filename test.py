from neural_network import NNLearner
import data_service
from sklearn.metrics import accuracy_score
import numpy as np

scale_data = True
transform_data = False
random_slice = None
random_seed = 777
dataset = 'breast_cancer'
test_size = 0.5

nn_activation = 'relu'
alpha = 0.0001
nn_hidden_layer_sizes = (10,)
nn_learning_rate = 'constant'
nn_learning_rate_init = 0.01
nn_solver = 'lbfgs'

#{'activation': 'relu', 'alpha': 0.0001, 'hidden_layer_sizes': (100,), 'learning_rate_init': 0.01, 'solver': 'lbfgs'}

nn_learner = NNLearner(hidden_layer_sizes=nn_hidden_layer_sizes, max_iter=200, solver=nn_solver, activation=nn_activation,
                       alpha=alpha, learning_rate=nn_learning_rate, learning_rate_init=nn_learning_rate_init)

x_train, x_test, y_train, y_test = data_service.load_and_split_data(scale_data=scale_data,
                                                                    transform_data=transform_data,
                                                                    random_slice=random_slice, random_seed=random_seed,
                                                                    dataset=dataset, test_size=test_size)

nn_accuracy_score, nn_fit_time, nn_predict_time = nn_learner.fit_predict_score(x_train, y_train, x_test, y_test)


print('nn: {0}, {1}, {2}'.format(nn_accuracy_score, nn_fit_time, nn_predict_time))
print('---------------------------------------------------------------')
print(nn_learner.estimator.coefs_)
print('---------------------------------------------------------------')
print(nn_learner.estimator.coefs_[0].shape)
print(nn_learner.estimator.coefs_[1].shape)
print(nn_learner.estimator.n_outputs_)

print("===========Steal the weights and set them on a new Learner:")

nn_learner_stolen_weights = NNLearner(hidden_layer_sizes=nn_hidden_layer_sizes, max_iter=200, solver=nn_solver, activation=nn_activation,
                       alpha=alpha, learning_rate=nn_learning_rate, learning_rate_init=nn_learning_rate_init)

#nn_learner_stolen_weights.estimator.coefs_ = nn_learner.estimator.coefs_
nn_learner_stolen_weights.estimator.coefs_ = [np.random.rand(30,10), np.random.rand(10,1)]
nn_learner_stolen_weights.estimator.n_outputs_ = nn_learner.estimator.n_outputs_
nn_learner_stolen_weights.estimator.n_layers_ = nn_learner.estimator.n_layers_
nn_learner_stolen_weights.estimator.intercepts_ = nn_learner.estimator.intercepts_
nn_learner_stolen_weights.estimator.out_activation_ = nn_learner.estimator.out_activation_
nn_learner_stolen_weights.estimator._label_binarizer = nn_learner.estimator._label_binarizer

#print(dir(nn_learner_stolen_weights.estimator))
prediction = nn_learner_stolen_weights.estimator.predict(x_test)

stolen_accuracy = accuracy_score(y_test, prediction)

print('STOLEN prediction: ', prediction[:20])
print('actual    : ', y_test[:20])
print('stolen_accuracy: ', stolen_accuracy)
print('-----------------------------------------------')




