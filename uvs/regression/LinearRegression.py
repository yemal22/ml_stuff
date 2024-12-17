import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.001, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.costs = []
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features).reshape(-1, 1)
        self.bias = 0

        for _ in range(self.n_iterations):
            predictions = np.dot(X, self.weights) + self.bias
            errors = predictions - y

            gradient_weights = (1 / n_samples) * np.dot(X.T, errors)
            gradient_bias = (1 / n_samples) * np.sum(errors)

            self.weights -= self.learning_rate * gradient_weights
            self.bias -= self.learning_rate * gradient_bias
            cost = np.mean(errors ** 2)

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
    