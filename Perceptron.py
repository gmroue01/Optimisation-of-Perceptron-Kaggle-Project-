
import scipy.io
import numpy as np
from scipy.sparse import vstack
class Perceptron_GD:

  def __init__(self,learning_rate=0.01,n_iters=100):
    self.lr = learning_rate
    self.n_iters = n_iters
    self.W = None
    self.bias = None
    self.loss_history = []

  def activation_func(self,z):
    #Sigmoide
    return (1+np.exp(z))**(-1)

  def mse_loss(self,y_pred,y_true):
    return np.mean((y_true - y_pred)**2)

  def oracle(self,X,y_true):
    n = X.shape[0]

    residuals = self.W @ X - y_true

    gradient_W = (1/n)*(residuals@X.T)

    gradient_B = (1/n)*sum(residuals)


    return gradient_W, gradient_B



  def fit(self,X,y):

      n_features = X.shape[0]


      self.W = np.ones((1,n_features))
      self.bias = 0.0



      for k in range(self.n_iters):

        y_pred = (self.W@X) + self.bias
        l = self.mse_loss(y_pred,y)
        self.loss_history.append(l)
        grad_W, grad_B = self.oracle(X,y)
        self.W = self.W - self.lr * grad_W
        self.bias = self.bias - self.lr*grad_B

  def predict(self,X):
    linear_product = self.W @ X + self.bias

    return np.where(linear_product >= 0, 1,0)




# model = Perceptron_GD(learning_rate=0.001)

# model.fit(X,y_ts)
# plt.plot(model.loss_history)
