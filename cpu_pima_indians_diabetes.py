import csv
import numpy as np

from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

from urllib import request

import cpu

database_url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv'
response = request.urlopen(database_url)

data = np.array(list(csv.reader(map(lambda x: x.decode('utf-8'), response.readlines()))), dtype=float)
X, y = data[:, 0:-1], data[:, -1]


kf = KFold(random_state=1, shuffle=True)

function_scores = []

functions_to_run = ['gradient', 'coordinate', 'stochastic_coordinate']

for function in functions_to_run:
    # Make model pipeline
    model = Pipeline(
        steps=[
            ("scale", MinMaxScaler()),
            ("regression", cpu.RidgeRegression(function=function)),
        ]
    )

    scores = []
    for train_index, test_index in kf.split(X):
        X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
        model.fit(X_train, y_train, regression__learning_rate=0.004, regression__epochs=500)
        scores.append(model.score(X_test, y_test))

    function_scores.append(np.mean(scores))


for function, score in zip(functions_to_run, function_scores):
    print(f"{function} function score:\t{score}")
