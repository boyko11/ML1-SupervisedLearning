from sklearn.metrics import accuracy_score

class Learner:

    def fit_predict_score(self, learner, x_train, y_train, x_test, y_test):

        learned_model = learner.fit(x_train, y_train)
        prediction = learned_model.predict(x_test)
        return accuracy_score(y_test, prediction)