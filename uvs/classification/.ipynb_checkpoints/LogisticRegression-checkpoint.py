import numpy as np

def sigmoid(z):
    # Calcul stable de la sigmoïde
    return np.where(
        z >= 0,
        1 / (1 + np.exp(-z)),           # pour les valeurs positives
        np.exp(z) / (1 + np.exp(z))     # pour les valeurs négatives
    )

class LogisticRegression:

    def __init__(self, n_iterations=1000, learning_rate=0.001):
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.weight = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weight = np.zeros(n_features).reshape(-1, 1)
        self.bias = 0
        for _ in range(self.n_iterations):
            linear_model = np.dot(X, self.weight) + self.bias
            y_predicted = sigmoid(linear_model)
            dw = (1/n_samples)*np.dot(X.T, (y_predicted - y))
            db = (1/n_samples)*np.sum(y_predicted - y)
            self.weight -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def predict(self, X):
        linear_model = np.dot(X, self.weight) + self.bias
        y_predicted = sigmoid(linear_model)
        return [0 if y <= 0.5 else 1 for y in y_predicted]