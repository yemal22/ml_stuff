import numpy as np

def eucledean_dist(x, y):
    distances = np.sqrt(np.sum((x-y)**2))
    return distances

class KNN:
    
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        return [self._predict(x) for x in X]

    def _predict(self, x):
        distances = [eucledean_dist(x, x_train) for x_train in self.X_train]
        neighbors_index = np.argsort(distances)[:self.k]
        neighbors_labels = [self.y_train[i] for i in neighbors_index]
        return np.argmax(np.bincount(neighbors_labels))