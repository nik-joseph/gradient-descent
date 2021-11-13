from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline


import datasets
from models.ridge_regression import RidgeRegression, GPURidgeRegression


model = Pipeline(
        steps=[
            ("scale", MinMaxScaler()),
            ("regression", RidgeRegression(batch_size=50)),
        ]
    )


X, y = datasets.get_pima_indian_diabetes()
X_train, X_test, y_train, y_test = train_test_split(X, y)

model = model.fit(
    X_train, y_train, regression__epochs=1000, regression__learning_rate=0.01, regression__l2=0.01
)
score = model.score(X_test, y_test)

print(score)

