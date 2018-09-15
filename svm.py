from sklearn import svm
from data_service import load_and_split_data
from accuracy_report_service import report_accuracy
from predict_service import predict

accuracy_sum = 0
num_runs = 10
for i in range(num_runs):

    X_train, X_test, Y_train, Y_test = load_and_split_data()

    #mlp_classifier = neural_network.MLPClassifier(hidden_layer_sizes=(64,64))
    mlp_classifier = svm.LinearSVC(random_state=77)

    prediction = predict(mlp_classifier, X_train, Y_train, X_test)

    accuracy = report_accuracy(Y_test, prediction)
    accuracy_sum += accuracy

print("Mean accuracy: {0}".format(accuracy_sum/float(num_runs)))