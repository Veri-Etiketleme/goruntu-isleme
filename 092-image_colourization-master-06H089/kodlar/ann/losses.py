from abc import ABC
import numpy as np


class Loss(ABC):
    def __init__(self):
        self.eval = None
        self.prime = None

    def loss(self, out_hat, out):
        if out_hat.shape != out.shape:
            raise Exception(f"y_hat({out_hat.shape}) and y({out.shape}) have different shapes")
        return np.mean(self.eval(out_hat, out), axis=0).squeeze()

    def grad(self, out_hat, out):
        if out_hat.shape != out.shape:
            raise Exception(f"y_hat({out_hat.shape}) and y({out.shape}) have different shapes")
        return np.mean(self.prime(out_hat, out) / out.size, axis=0).squeeze()


class MSE(Loss):
    """
    || y_hat - y ||
    returns (size, y_dim)
    """
    def __init__(self):
        super().__init__()
    
    def loss(self, y_hat, y):
        return np.sum((y_hat - y)**2, axis=1)
    
    def grad(self, y_hat, y):
        return 2 * (y_hat - y)


    
# Not Working, check why    
class BCE(Loss):
    def __init__(self):
        super().__init__()
        
    def loss(self, y_hat, y):
        return np.mean(-1*(y * np.log(y_hat) + (1-y)*np.log(1-y_hat)), axis=0)
    
    def grad(self, y_hat, y):
        return np.mean(-1 * (y - y_hat)/(y_hat * (1 - y_hat)), axis=0)
#         return y_hat - y
        

class CCE(Loss):
    def __init__(self):
        super(CCE, self).__init__()

    def loss(self, out_hat, out):
        return np.mean(-np.sum(np.multiply(out, np.log(out_hat)), axis=1))

    def grad(self, out_hat, out):
        return np.mean(out_hat)


if __name__ == "__main__":
    import numpy as np
    y_hat = np.array([1, 3, 5])
    y = np.array([1, 5, 9])
    mse = MSE()
    print(mse.loss(y_hat, y), mse.grad(y_hat, y))
