from typing import Dict, Tuple 
import numpy as np 
from utils.metrics import mean_squared_error
import math


class LinearRegression:
    """Linear model.

    Args:
        n_iterations (float): The number of training iterations the algorithm will tune the weights for.
        learning_rate (float): The step length that will be used when updating the weights.
        gradient_descent (boolean): True or false depending if gradient descent should be used when training. If 
        false then we use batch optimization by least squares.
    """

    def __init__(self, n_iterations:int=100,
                 lr:float=0.01):
       self.n_iterations = n_iterations
       self.lr = lr 
       self.W = None
       self.loss_over_time = []

    def initialize_weights(self, n_features):
        """Initialize weights randomly 

        Args:
            n_features (int): Number of features in X
        """
        limit = 1 / math.sqrt(n_features)
        self.W = np.random.uniform(-limit, limit, (n_features, ))

    def fit(self, 
            X_train: np.ndarray, 
            y_train: np.ndarray):

        # Initialize randomly W
        self.initialize_weights(X_train.shape[1])

        self.loss_over_time= []
        self.w_over_time = []
        
        # Do Gradient Descent
        for _ in range(self.n_iterations):
            y_hat = np.dot(self.W, X_train.T)
            self.loss_over_time.append(mean_squared_error(y_train, y_hat))
            # Weights are updatated by - lr(error).dot X_train
            error = y_hat - y_train
            w_grad = (2 / X_train.shape[0])* X_train.T.dot(error)
            self.W -= self.lr * w_grad
            self.w_over_time.append(float(self.W))

        
    def predict(self, 
                X_test: np.ndarray, 
                W: np.ndarray= None, 
                bias: np.array=0 ):
        
        if W is None: W = self.W 
        if self.W is None: raise ValueError("A Weight array must be passed or `.fit()` before")
        self.initialize_weights(X_test.shape[1])
        return np.dot(X_test, W) + bias

class RidgeRegression:
    """Linear model.

    Args:
        n_iterations (float): The number of training iterations the algorithm will tune the weights for.
        learning_rate (float): The step length that will be used when updating the weights.
        reg_factor (float): Regularization factor for the ridge regression
    """

    def __init__(self, n_iterations:int=100,
                 lr:float=0.01, 
                 reg_factor:float=0.02):
       self.n_iterations = n_iterations
       self.lr = lr 
       self.W = None
       self.reg_factor = reg_factor
       assert reg_factor >= 0, "Reg factor must be a positive value"
       self.loss_over_time = []

    def initialize_weights(self, n_features):
        """Initialize weights randomly 

        Args:
            n_features (int): Number of features in X
        """
        limit = 1 / math.sqrt(n_features)
        self.W = np.random.uniform(-limit, limit, (n_features, ))

    def fit(self, 
            X_train: np.ndarray, 
            y_train: np.ndarray):

        # Initialize randomly W
        self.initialize_weights(X_train.shape[1])

        self.loss_over_time= []
        self.w_over_time = []
        # Do gradient_descent:
        for _ in range(self.n_iterations):
            y_hat = np.dot(self.W, X_train.T)
            self.loss_over_time.append(mean_squared_error(y_train, y_hat))
            # Weights are updatated by - lr(error).dot X_train
            error = y_hat - y_train
            w_grad = (2 / X_train.shape[0])* X_train.T.dot(error)  + 2*self.reg_factor*(self.W.T)
            self.W -= self.lr * w_grad
            self.w_over_time.append(self.W.copy())

        
    def predict(self, 
                X_test: np.ndarray, 
                W: np.ndarray= None, 
                bias: np.array=0 ):
        
        if W is None: W = self.W 
        if self.W is None: raise ValueError("A Weight array must be passed or `.fit()` before")
        self.initialize_weights(X_test.shape[1])
        return np.dot(X_test, W) + bias
