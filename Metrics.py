import numpy as np


class Metrics:

    def __init__(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        self.size = len(y_true)
        self.accuracy = None
        self.mse_loss = None
        self.recall = None
        self.precision = None

        # Defintion des TP,FP,FN,FP

        self.TP = np.sum((self.y_true == 1) & (self.y_pred == 1))
        self.FP = np.sum((self.y_true == 0) & (y_pred == 1))
        self.FN = np.sum((self.y_true == 1) & (self.y_pred == 0))
        self.TN = np.sum((self.y_true == 0) & (self.y_pred == 0))

    def accuracy_score(self):
        self.accuracy = (self.TP + self.TN)/(self.size)

    def mse_loss_scores(self):
        self.mse_loss = np.mean((self.y_true - self.y_pred)**2)

    def recall_score(self):
        self.recall = (self.TP)/(self.TP + self.TN)

    def precision_score(self):
        self.precision = self.TP/(self.TP + self.FP)


y_true = np.concatenate((np.zeros((1, 50)), np.zeros((1, 50))), axis=1)
y_pred = np.concatenate((np.ones((1, 50)), np.ones((1, 50))), axis=1)


m = Metrics(y_pred=y_pred, y_true=y_true)
m.recall_score()
print(m.recall)
