import pandas as panda
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate
import data_prep
from sklearn.metrics import precision_score, recall_score
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
import numpy as np


def print_metrics(scores, labels, precision, recall):
    '''
    This method prints out a list of evaluation metrics retrieved from a pipeline
    :param scores: Dictionary holding accuracy and balanced accuracy
    :param confusion: A confusion matrix for each class
    :param labels: The set of categories
    :param precision: The average precision
    :param recall: The average recall
    '''
    print("Accuracy: ", np.mean(scores['test_acc']))
    print("Balanced Accuracy: ", np.mean(scores['test_bal_acc']))
    print("Precision: ", precision)
    print("Recall: ", recall)

COLOR = "color"
colors = ['beige', 'black', 'blue', 'brown', 'gray', 'green', 'orange', 'pink', 'purple', 'red', 'white', 'yellow']

# Data pre-processing
df = panda.read_csv('data_train.csv')
df = data_prep.clean_bad_rows(df)
y = df[COLOR]
x = data_prep.remove_columns(df)

scoring = {'acc': 'accuracy', 'bal_acc': 'balanced_accuracy'}
# Create pipeline
lg_pl = make_pipeline(preprocessing.StandardScaler(),
                      LogisticRegression( class_weight="balanced", max_iter=10000))
# Evaluate model
scores = cross_validate(lg_pl, x, y, scoring=scoring, cv=4)
y_pred = cross_val_predict(lg_pl, x, y, cv=4)
precision = precision_score(y, y_pred, average='macro')
recall = recall_score(y, y_pred, average='macro')
print_metrics(scores, colors, precision, recall)


