import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from tqdm import tqdm

def initialisation(X:np.array):
    W = np.random.randn(X.shape[1], 1)
    b = np.random.randn(1)
    return (W, b)

def model(X, W, b):
    Z = X.dot(W) + b
    Z = np.clip(Z, -50000, 50000)
    A = 1 / (1 + np.exp(-Z))
    return A

def log_loss(A, y):
    epsilon = 1e-15  # Petite valeur pour éviter les log(0)
    A = np.clip(A, epsilon, 1 - epsilon)
    L = 1/len(y) * np.sum(-y * np.log(A) - (1 - y) * np.log(1 - A))
    return L

def gradients(A, X, y):
    dw = 1/len(y) * np.dot(X.T, A - y)
    db = 1/len(y) * np.sum(A - y)
    return (dw, db)

def update(dw, db, W, b, learning_rate):
    W = W - learning_rate*dw
    b = b - learning_rate*db
    return (W, b)

def predict(X, W, b):
    A = model(X, W, b)
    return (A >= 0.5).astype(int)

def artificial_neuron(X, y, learning_rate=0.01, n_iterations=1000):
    W , b = initialisation(X)
    
    Loss = []
    acc = []
    for i in tqdm(range(n_iterations)):
        A = model(X, W, b)
        
        if i % 10 == 0:
            Loss.append(log_loss(A, y))
            y_pred = predict(X, W, b)
            acc.append(accuracy_score(y, y_pred))
        
        dw, db = gradients(A, X, y)
        W, b = update(dw, db, W, b, learning_rate)
        
    
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(Loss)
    plt.title('Loss')
    plt.subplot(1, 2, 2)
    plt.plot(acc)
    plt.title('Accuracy')
    plt.show()
    return (W, b)