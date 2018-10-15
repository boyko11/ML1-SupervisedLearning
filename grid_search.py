#http://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_digits.html#sphx-glr-auto-examples-model-selection-plot-grid-search-digits-py
from __future__ import print_function

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import data_service

print(__doc__)

scale_data = True
transform_data = True
random_slice = 10000
random_seed = 777
dataset="kdd"
test_size=0.5

x_train, x_test, y_train, y_test = data_service.load_and_split_data(scale_data=scale_data,
                                                                    transform_data=transform_data,
                                                                    random_slice=random_slice, random_seed=random_seed,
                                                                    dataset=dataset, test_size=test_size)

# Set the parameters by cross-validation
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [0.001, 0.0001],
                     'C': [1, 10, 100, 1000], 'class_weight': ['balanced', None]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000], 'class_weight': ['balanced', None]}]

# neural_network.MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, solver=solver,
#             activation=activation, alpha=alpha, random_state=1, max_iter=max_iter, learning_rate=learning_rate,
#                                                       learning_rate_init=learning_rate_init)
tuned_parameters = [{
                     'solver': ['lbfgs'],
                     'learning_rate_init': [0.0001, 0.01, 1],
                     'hidden_layer_sizes': [(100,)],
                     'activation': ['relu'],
                     'alpha': [0.0001, 0.01, 1]
                    },
                    {
                     'solver': ['sgd'],
                     'learning_rate': ['constant', 'invscaling', 'adaptive'],
                     'learning_rate_init': [0.0001, 0.01, 1],
                     'hidden_layer_sizes': [(100,)],
                     'activation': ['relu'],
                     'alpha': [0.0001, 0.01, 1]
                    }
]

estimator = SVC();
estimator = MLPClassifier()

scores = ['precision_macro', 'recall_macro', 'accuracy']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(estimator, tuned_parameters, cv=5,
                       scoring=score, n_jobs=-1)
    clf.fit(x_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_pred = clf.predict(x_test)
    print(classification_report(y_test, y_pred))
    print()