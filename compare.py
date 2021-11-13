from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from urllib import request
import matplotlib.pyplot as plt

import numpy as np

import csv
import cpu
import gpu


database_url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv'
response = request.urlopen(database_url)

data = np.array(list(csv.reader(map(lambda x: x.decode('utf-8'), response.readlines()))), dtype=float)
X, y = data[:, 0:-1], data[:, -1]


models = [
    Pipeline(
        steps=[
            ("scale", MinMaxScaler()),
            ("regression", model_type.RidgeRegression(function='stochastic_coordinate')),
        ]
    )
    for model_type in [cpu, gpu]
]

X_train, X_test, y_train, y_test = train_test_split(X, y)


reports = [
    model.fit(
            X_train, y_train, regression__learning_rate=0.01, regression__epochs=1000, regression__analysis=True
    ).steps[1][1].report

    for model in models
]

reports = list(map(lambda a: list(map(lambda x: (x[1].total_seconds(), 1 - x[0]), a)), reports))

reports = list(map(lambda x: list(zip(*x)), reports))

for report, label in zip(reports, ['cpu', 'gpu']):
    plt.plot(report[0], report[1], label=label)

plt.xlabel('Execution time(seconds)')
plt.ylabel('Error')
plt.legend()
plt.show()
