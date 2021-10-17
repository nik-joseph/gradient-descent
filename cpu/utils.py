import numpy as np


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def sigmoid_derivative(y_hat):
    return y_hat * (1 - y_hat)


def predict(x, w, b):
    """
    :param x: input row
    :param w: model weights
    :param b: bias
    :return: prediction y hat
    """
    x_ar, w_ar = np.array(x, dtype=float), np.array(w, dtype=float)
    if x_ar.shape != w_ar.shape:
        raise Exception("Input and weight not same size in prediction")
    return sigmoid(b + np.dot(x_ar, w_ar.transpose()))


def stochastic_gradient(train_data_X, train_data_y, learning_rate=None, epochs=None):
    """
    :param train_data_X: input data attributes
    :param train_data_y: input data y hat
    :param learning_rate: learning rate
    :param epochs: number of iterations to train
    :return: weight and bias of the model w, b
    """
    # Initialize weight and bias to 0
    w, b = np.zeros(len(train_data_X[0])), 0

    # Raise required errors
    if not learning_rate:
        raise Exception("Learning rate not set!")
    if not epochs:
        raise Exception("Number of epochs not set!")

    # Start epoch
    for epoch in range(epochs):
        for input_X, input_y in zip(train_data_X, train_data_y):
            # Compute prediction
            y_hat = predict(input_X, w, b)

            # Calculate error
            error = input_y - y_hat

            # Update weight and bias on learning rate and calculated error
            b = b + learning_rate * error * sigmoid_derivative(y_hat)
            w = w + learning_rate * error * sigmoid_derivative(y_hat) * input_X
    return w, b
