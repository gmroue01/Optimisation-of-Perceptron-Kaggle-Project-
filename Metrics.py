import numpy as np


class Metrics:

    def __init__(self, y_pred, y_true):
        self.y_pred = np.array(y_pred)
        self.y_true = np.array(y_true)

        # --- CORRECTION FINALE : Adaptation à ton format (Classes, Échantillons) ---
        
        # y_true est de taille (20, 2792). On cherche l'index max sur l'axe 0 (les colonnes)
        if self.y_true.ndim > 1:
            self.y_true_classes = np.argmax(self.y_true, axis=0)
        else:
            self.y_true_classes = self.y_true.flatten()

        # y_pred sort déjà de ta fonction "predict" en format (2792,) grâce à ton propre argmax
        if self.y_pred.ndim > 1:
            self.y_pred_classes = np.argmax(self.y_pred, axis=0)
        else:
            self.y_pred_classes = self.y_pred.flatten()
        # -------------------------------------------------------------------------

        # Sécurité
        if self.y_true_classes.shape != self.y_pred_classes.shape:
            raise ValueError(f"Erreur de dimension : y_true {self.y_true_classes.shape} vs y_pred {self.y_pred_classes.shape}")

        self.size = len(self.y_true_classes)
        
        # On s'assure que y_true_classes commence bien à 0 ou 1 pour correspondre
        self.classes = np.unique(np.concatenate((self.y_true_classes, self.y_pred_classes)))

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
