import warnings
import pandas as panda
from sklearn.metrics import balanced_accuracy_score
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.preprocessing import StandardScaler
import data_prep
from sklearn.model_selection import train_test_split
from sklearn import svm

COLOR = "color"
TEXTURE = "texture"

# Data preprocessing
df = panda.read_csv('data_train.csv')
df = data_prep.clean_bad_rows(df)
# Set category
y = df[TEXTURE]
x = data_prep.remove_columns(df)

# Split data into test an train
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.66, random_state=9)

warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning)
warnings.filterwarnings(action='ignore', category=UserWarning)

# Define parameter sets
gamma = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]
C = [0.01, 0.1, 1, 10, 100, 1000, ]
decision_function_shape = ['ovo', 'ovr']
weightings = ['balanced', None]

# Scale data without leakage
testScaler = StandardScaler()
testScaler.fit(X_train)
X_train = testScaler.transform(X_train)
X_test = testScaler.transform(X_test)

score = 0
# Loop over each set of parameters
for g in gamma:
    for i in C:
        for d in decision_function_shape:
            for weight in weightings:
                # Create new model
                model = svm.SVC(kernel="rbf", decision_function_shape=d, class_weight=weight, C=i, gamma=g)
                # Train Model
                model.fit(X_train, y_train)
                # Calculate and output score
                prediction = model.predict(X_test)
                ba = balanced_accuracy_score(y_test, prediction)
                if ba > score:
                    print("Gamma: ", g, " C: ", i, " DFS: ", d, " Class_weight: ", weight, )
                    print(ba)
                    score = ba
