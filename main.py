
import Perceptron_GD

if __name__ == "__main__":
    
    data = scipy.io.loadmat("data_doc.mat",squeeze_me=True,struct_as_record=False)
    print(data.keys())

    # Input
    X = data['Xts'].tocsr()
    # Label
    y_ts = data['yts']
    #Validation
    X_vr = data['Xvr'].tocsr()

    # Definition of index
    n = X.shape[0]
    d = X.shape[1]


