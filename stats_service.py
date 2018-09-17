import numpy as np


def record_stats(accuracy_scores, accuracy_score, fit_times, fit_time, predict_times, predict_time):
    accuracy_scores.append(accuracy_score)
    fit_times.append(fit_time)
    predict_times.append(predict_time)


def mean(python_list):

    return np.mean(np.asarray(python_list))
