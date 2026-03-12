
import scipy.io
import numpy as np
from scipy.sparse import vstack
from Perceptron import Perceptron_GD
from Data import DataPreprocessing
import matplotlib.pyplot as plt
from OptimalStep import puissance
from Metrics import Metrics
from sklearn.model_selection import train_test_split


if __name__ == "__main__":

    d = DataPreprocessing("./data/data_doc.mat", seed=42)

    X_train, y_train, X_test, y_test = d.fit()
    # Calcul de Hessienne

    X_train, X_test = d.transform()
    # Calcul de la valeur propre maximum de H
    # print("Calcul de la valeur propre de module maximum")
    # L = puissance(H)
    # print(f"La valeur propre de module maximum vaut {L}")

    model = Perceptron_GD(
        input_size=X_train.shape[0], output_size=20, learning_rate=1e-3, n_iters=10)
    model.fit(X_train, y_train)

    # # Plot the loss history
    plt.plot(model.loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss History")
    plt.savefig("./results/loss_history.png")
    print("Loss history saved to loss_history.png")

    y_pred = model.predict(X_test)
    metrics = Metrics(y_pred=y_pred, y_true=y_test)

    metrics.transform_metrics()
