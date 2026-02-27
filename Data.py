from sklearn.preprocessing import MaxAbsScaler
import scipy.io
from sklearn.model_selection import train_test_split
class DataPreprocessing :

    def __init__(self, filepath,seed):
        self.file = filepath
        self.data = scipy.io.loadmat("./data/data_doc.mat",
                            squeeze_me=True, struct_as_record=False)
        self.X = self.data['Xts'].tocsr()
        self.y_ts = self.data['yts']
        self.X_vr = self.data['Xvr'].tocsr()
        self.seed = seed
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None



    def fit(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
        self.X, self.y_ts, test_size=0.2, random_state=self.seed)

        return self.X_train,self.y_train,self.X_test,self.y_test
    


    
