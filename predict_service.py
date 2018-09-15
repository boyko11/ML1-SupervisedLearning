
def predict(learner, X_train, Y_train, X_test):

    learned_model = learner.fit(X_train, Y_train)
    prediction = learned_model.predict(X_test)
    return prediction