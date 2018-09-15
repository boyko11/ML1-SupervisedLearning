import numpy as np
from sklearn.metrics import accuracy_score


def report_accuracy(Y_test, prediction):

    training_records_correctly_clasified = np.logical_not(np.logical_xor(Y_test, prediction))
    accuracy = np.sum(training_records_correctly_clasified) / len(training_records_correctly_clasified)

    # print('accuracy rate: {0}'.format(accuracy))
    # print('error rate: {0}'.format(1 - accuracy))
    accuracy = accuracy_score(Y_test, prediction)
    print(accuracy)

    return accuracy
