
import scipy.io
import numpy as np
from scipy.sparse import vstack
from Perceptron import Perceptron_GD
import matplotlib.pyplot as plt

if __name__ == "__main__":
    
    data = scipy.io.loadmat("data_doc.mat",squeeze_me=True,struct_as_record=False)
    # Input
    X = data['Xts'].tocsr()
    # Label
    y_ts = data['yts']
    #Validation
    X_vr = data['Xvr'].tocsr()


    model = Perceptron_GD(learning_rate=0.01,n_iters=100)
    model.fit(X,y_ts)

    # Plot the loss history
    plt.plot(model.loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss History")
    plt.savefig("loss_history.png")
    print("Loss history saved to loss_history.png")




