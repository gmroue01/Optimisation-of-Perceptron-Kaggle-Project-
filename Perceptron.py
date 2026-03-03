
import numpy as np


class Perceptron_GD:

    def __init__(self, input_size, output_size, learning_rate=0.01, n_iters=100):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.input_size = input_size
        self.output_size = output_size
        self.W = None
        self.bias = None
        self.loss_history = []

    def activation_func(self, z):

        return 1/(1+np.exp(-z))

    def deriative_sigmoid(self, z):
        sig = self.activation_func(z)
        return sig*(1-sig)

    def mse_loss(self, y_pred, y_true):
        return np.mean((y_true - y_pred)**2)

    def oracle(self, X, y_true):
        n = X.shape[1]

        residuals = self.W.T@X + self.bias
        y_pred = self.activation_func(residuals)

        dL_dy = 2*(y_pred - y_true)

        dy_dz = self.deriative_sigmoid(residuals)

        Error_Sigmoid = (1/n*(dL_dy*dy_dz))

        gradient_W = X@Error_Sigmoid.T

        gradient_B = np.sum(Error_Sigmoid, axis=1, keepdims=True)

        return gradient_W, gradient_B, y_pred

    def fit(self, X, y_true):

        self.W = np.random.randn(self.input_size, self.output_size)
        self.bias = np.random.randn(self.output_size, 1)

        for k in range(self.n_iters):
            grad_W, grad_B, y_pred = self.oracle(X, y_true)

            loss = self.mse_loss(y_pred, y_true)
            self.loss_history.append(loss)

            self.W = self.W - self.lr * grad_W
            self.bias = self.bias - self.lr*grad_B

    def predict(self, X):
        print("predict X", X.shape)
        print("W :", self.W.shape)
        print("bias : ", self.bias.shape)
        linear_product = self.W.T @ X + self.bias
        return np.argmax(linear_product, axis=0)

        # model = Perceptron_GD(learning_rate=0.001)

        # model.fit(X,y_ts)
        # plt.plot(model.loss_history)
