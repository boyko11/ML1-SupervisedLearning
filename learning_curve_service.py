import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

#http://scikit-learn.org/0.15/auto_examples/plot_learning_curve.html
def plot_learning_curve(estimator, estimator_non_scaled, title, X, y, X_non_scaled, Y_non_scaled, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 10), draw_non_scaled=False):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Error")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

    train_errors = 1 - train_scores
    test_errors = 1 - test_scores

    train_errors_mean = np.mean(train_errors, axis=1)
    train_errors_std = np.std(train_errors, axis=1)
    test_errors_mean = np.mean(test_errors, axis=1)
    test_errors_std = np.std(test_errors, axis=1)

    print('train_errors_mean:')
    print(train_errors_mean)

    plt.grid()

    plt.fill_between(train_sizes, train_errors_mean - train_errors_std,
                     train_errors_mean + train_errors_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_errors_mean - test_errors_std,
                     test_errors_mean + test_errors_std, alpha=0.1, color="g")

    plt.plot(train_sizes, train_errors_mean, 'o-', color="r",
             label="Training error")
    plt.plot(train_sizes, test_errors_mean, 'o-', color="g",
             label="Test error")

    if draw_non_scaled:
        train_sizes_non_scaled, train_scores_non_scaled, test_scores_non_scaled = learning_curve(
            estimator_non_scaled, X_non_scaled, Y_non_scaled, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

        train_errors_non_scaled = 1 - train_scores_non_scaled
        test_errors_non_scaled = 1 - test_scores_non_scaled
        train_errors_non_scaled_mean = np.mean(train_errors_non_scaled, axis=1)
        test_errors_non_scaled_mean = np.mean(test_errors_non_scaled, axis=1)

        plt.plot(train_sizes, train_errors_non_scaled_mean, color="r", linestyle='dashed',
                 label="Training error Non-Scaled")
        plt.plot(train_sizes, test_errors_non_scaled_mean, color="g", linestyle='dashed',
                 label="Test error Non-Scaled")

    plt.legend(loc="best")
    #plt.show()
    plt.savefig('{0}.png'.format('_'.join(title.split())))
