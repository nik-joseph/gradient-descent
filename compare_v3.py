import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline


import datasets
from models.ridge_regression import RidgeRegression, GPURidgeRegression

reports = []
batch_size = 50
for required_model in [RidgeRegression, GPURidgeRegression]:
    model = Pipeline(
            steps=[
                ("scale", MinMaxScaler()),
                ("regression", required_model(batch_size=batch_size, benchmark=True)),
            ]
        )

    X, y = datasets.get_pima_indian_diabetes()
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    model = model.fit(
        X_train, y_train, regression__epochs=3000, regression__learning_rate=0.01, regression__l2=0.01
    )

    actual_model, scale = model.steps[1][1], model.steps[0][1]
    reports.append(actual_model.get_benchmark_results(scale.transform(X_test), y_test))
    print('********************************************************************************')

reports = map(lambda r: map(lambda item: (item[0], 1 - item[1]), r), reports)
reports = list(map(lambda x: list(zip(*x)), reports))

for report, label in zip(reports, ['cpu', 'gpu']):
    plt.plot(report[0], report[1], label=label)

plt.xlabel('Execution time(seconds)')
plt.ylabel('Error')
plt.title(f"Batch size {batch_size}")
plt.legend()
plt.show()
