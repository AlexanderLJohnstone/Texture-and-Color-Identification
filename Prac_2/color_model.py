import csv
import pandas as panda
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate
import data_prep
from sklearn.metrics import precision_score, recall_score
from sklearn import svm, preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict
import numpy as np


def print_metrics(scores, confusion, labels, precision, recall):
    '''
    This method prints out a list of evaluation metrics retrieved from a pipeline
    :param scores: Dictionary holding accuracy and balanced accuracy
    :param confusion: A confusion matrix for each class
    :param labels: The set of categories
    :param precision: The set of precision values
    :param recall: The set of recall values
    '''
    print("Accuracy: ", np.mean(scores['test_acc']))
    print("Balanced Accuracy: ", np.mean(scores['test_bal_acc']))
    for i in range(0, len(labels)):
        print(labels[i], " Precision")
        print(precision[i])
        print(labels[i], " Recall")
        print(recall[i])
    print("Confusion Matrix: ")
    print(confusion)


COLOR = "color"
colors = ['beige', 'black', 'blue', 'brown', 'gray', 'green', 'orange', 'pink', 'purple', 'red', 'white', 'yellow']

# Data pre-processing
df = panda.read_csv('data_train.csv')
df = data_prep.clean_bad_rows(df)
y = df[COLOR]
x = data_prep.remove_columns(df)

scoring = {'acc': 'accuracy', 'bal_acc': 'balanced_accuracy'}
# Create Pipeline
svm_pl = make_pipeline(preprocessing.StandardScaler(),
                       svm.SVC(kernel="rbf", decision_function_shape="ovo", class_weight='balanced', C=100,
                               gamma=1e-05))
# Evaluate Model
scores = cross_validate(svm_pl, x, y, scoring=scoring, cv=4)
y_pred = cross_val_predict(svm_pl, x, y, cv=4)
conf_mat = confusion_matrix(y, y_pred, labels=colors)
precision = precision_score(y, y_pred, average=None)
recall = recall_score(y, y_pred, average=None)
print_metrics(scores, conf_mat, colors, precision, recall)

# Load in test set
test_set = panda.read_csv('data_test.csv')
test_set = data_prep.remove_columns(test_set)
test_set = data_prep.fix_rows(test_set)

# Create new model
model = svm.SVC(kernel="rbf", decision_function_shape="ovo", class_weight='balanced', C=100, gamma=1e-05)

# Scale test set using training set scaler
scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)
test_set = scaler.transform(test_set)
# Train the model
model.fit(x, np.ravel(y.to_numpy().tolist()))
# Get predictions
prediction = model.predict(test_set)
#   Write predictions to CSV
with open('color_test.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    for each in prediction:
        writer.writerow([each])
