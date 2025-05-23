import numpy as np 

def mean_squared_error(y: np.ndarray, y_hat: np.ndarray) -> float:
        """Returns Mean squared error 

        Args:
            y (ndarray): Actual values 
            y_hat (ndarray): Predicted values 

        Returns:
            float: MSE 
        """
        
        assert y.shape == y_hat.shape, "Y and Y_hat must have the same shape"

        return np.mean(np.power(y_hat - y, 2)) 

def mean_averaged_error(y: np.ndarray, y_hat: np.ndarray):
    """
    Compute mean absolute error.

    Args:
        y (ndarray): Actual values 
        y_hat (ndarray): Predicted values 

    Returns:
        float: MAE
    """ 
    assert (y.shape == y_hat.shape).all(), "Y and Y_hat must have the same shape"
    return np.mean(np.abs(y_hat - y))

def root_mean_squared_error(y: np.ndarray, y_hat:np.ndarray) -> float:
    """Compute root mean squared error 

    Args:
        y (np.ndarray): Actual values 
        y_hat (np.ndarray): Predicted values

    Returns:
        float: _description_
    """
    return np.sqrt(np.mean(np.power(y_hat - y, 2)))
