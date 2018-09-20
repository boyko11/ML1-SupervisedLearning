from sklearn.metrics import accuracy_score
import time


class Learner:

    def fit_predict_score(self, learner, x_train, y_train, x_test, y_test):

        start_learning_time = time.time()
        learned_model = learner.fit(x_train, y_train)
        learning_time = time.time() - start_learning_time

        start_prediction_time = time.time()
        prediction = learned_model.predict(x_test)
        predicion_time = time.time() - start_prediction_time
        #print(accuracy_score(y_test, prediction), learner.score(x_test, y_test))

        return accuracy_score(y_test, prediction), learning_time, predicion_time