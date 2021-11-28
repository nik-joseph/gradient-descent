import csv
import numpy as np
from urllib import request


def get_pima_indian_diabetes():
    database_url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv'
    response = request.urlopen(database_url)
    data = np.array(list(csv.reader(map(lambda x: x.decode('utf-8'), response.readlines()))), dtype=float)
    X, y = data[:, 0:-1], data[:, -1]
    return X, y
