from sklearn import svm
from learner import Learner


class SVMLearner(Learner):

    def fit_predict_score(self, x_train, y_train, x_test, y_test):

        svm_classifier = svm.LinearSVC(random_state=2)

        return super(SVMLearner, self).fit_predict_score(svm_classifier, x_train, y_train, x_test, y_test)
