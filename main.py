
import scipy.io
import numpy as np
from scipy.sparse import vstack
from Perceptron import Perceptron_GD
import matplotlib.pyplot as plt
from OptimalStep import puissance
from Metrics import Metrics
from sklearn.model_selection import train_test_split

if __name__ == "__main__":

    data = scipy.io.loadmat("./data/data_doc.mat",
                            squeeze_me=True, struct_as_record=False)

    X = data['Xts'].tocsr()

    y_ts = data['yts']

    X_vr = data['Xvr'].tocsr()

    # labels = [i for i in range(20)]

    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y_ts, test_size=0.2, random_state=42, stratify=labels)

    labels = 20

    n = X.shape[1]
    # Calcul de Hessienne
    H = (1/n)*X@X.T

    # Calcul de la valeur propre maximum de H
    # print("Calcul de la valeur propre de module maximum")
    # L = puissance(H)
    # print(f"La valeur propre de module maximum vaut {L}")

    model = Perceptron_GD(
        input_size=X.shape[0], output_size=labels, learning_rate=1, n_iters=100)
    model.fit(X, y_ts)

    # Plot the loss history
    plt.plot(model.loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss History")
    plt.savefig("./results/loss_history.png")
    print("Loss history saved to loss_history.png")
