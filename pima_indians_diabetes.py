from cpu import LogisticRegression

import csv
from urllib import request
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline

database_url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv'
response = request.urlopen(database_url)

data = np.array(list(csv.reader(map(lambda x: x.decode('utf-8'), response.readlines()))), dtype=float)
X, y = data[:, 0:-1], data[:, -1]


kf = KFold(random_state=1, shuffle=True)

scores = []
model = make_pipeline(
    MinMaxScaler(),
    LogisticRegression()
)

for train_index, test_index in kf.split(X):
    X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
    model.fit(X_train, y_train, logisticregression__learning_rate=0.1, logisticregression__epochs=100)
    scores.append(model.score(X_test, y_test))

print(np.mean(scores))
