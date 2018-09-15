from sklearn import tree
from sklearn.datasets import load_breast_cancer
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


data = load_breast_cancer()
X = data.data
Y = data.target

X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size=0.2)

dt_classifier = tree.DecisionTreeClassifier()
dt_classifier_model = dt_classifier.fit(X_train, Y_train)

prediction = dt_classifier_model.predict(X_test)

print('prediction: {0}'.format(prediction))

training_records_correctly_clasified = np.logical_not(np.logical_xor(Y_test, prediction))
accuracy = np.sum(training_records_correctly_clasified) / len(training_records_correctly_clasified)

print('accuracy rate: {0}'.format(accuracy))
print('error rate: {0}'.format(1 - accuracy))

print(accuracy_score(Y_test, prediction))





