import numpy as np


class Metrics:

    def __init__(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true

        self.y_pred_classes = y_pred.flatten()
        self.y_true_classes = y_true.flatten()

        self.size = len(y_true)
        self.classes = np.unique(np.concatenate(
            (self.y_true_classes, self.y_pred_classes)))

        self.accuracy = None
        self.mse_loss = None
        self.recall = None
        self.precision = None

    def accuracy_score(self):
        correct_prediction = np.sum(self.y_true_classes == self.y_pred_classes)
        self.accuracy = correct_prediction / self.size

    def mse_loss_scores(self):
        self.mse_loss = np.mean((self.y_true - self.y_pred)**2)

    def precision_recall_scores(self):
        precisions = []
        recalls = []

        for c in self.classes:

            TP = np.sum((self.y_pred_classes == c) &
                        (self.y_true_classes == c))
            FP = np.sum((self.y_pred_classes == c) &
                        (self.y_true_classes != c))
            FN = np.sum((self.y_pred_classes != c) &
                        (self.y_true_classes == c))

            if (TP + FP) == 0:
                precisions.append(0.0)
            else:
                precisions.append(TP/(TP+FP))

            if ((TP+FN) == 0):
                recalls.append(0.0)
            else:
                recalls.append(TP / (TP+FN))

        self.precision = np.mean(precisions)
        self.recall = np.mean(recalls)

    def transform_metrics(self):
        print("----------------------------------------------")
        self.accuracy_score()
        print(f"Accuracy : {self.accuracy: .4f}\n")

        self.mse_loss_scores()
        print(f"MSE loss : {self.mse_loss: .4f}\n")
        self.precision_recall_scores()
        print(f"Recall : {self.recall: .4f}\n")
        print(f"Precision : {self.precision: .4f}\n")
        print("------------------------------------------")
