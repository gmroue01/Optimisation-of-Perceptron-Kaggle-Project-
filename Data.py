from sklearn.preprocessing import MaxAbsScaler
import scipy.io
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler


class DataPreprocessing:

    def __init__(self, filepath, seed):
        self.file = filepath
        self.data = scipy.io.loadmat("./data/data_doc.mat",
                                     squeeze_me=True, struct_as_record=False)
        self.X = self.data['Xts'].tocsr()
        self.y_ts = self.data['yts']
        self.X_vr = self.data['Xvr'].tocsr()

        self.nbr_documents = len(self.y_ts)

        self.classes = len(np.unique(self.y_ts))
        print(f"Nombre de documents : {self.nbr_documents}")
        print(f"Nombre de classes : {self.classes}")

        self.y_one_hot = np.zeros((self.classes, self.nbr_documents))
        for index, value in enumerate(self.y_ts.flatten()):

            self.y_one_hot[value - 1, index] = 1
        self.seed = seed
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

    def fit(self):

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X.T, self.y_one_hot.T, test_size=0.2, random_state=self.seed)

        self.X_train = self.X_train.T
        self.X_test = self.X_test.T

        self.y_train = self.y_train.T
        self.y_test = self.y_test.T

        return self.X_train, self.y_train, self.X_test, self.y_test

    def transform(self):

        # Définition de la matrice Y

        scaler = StandardScaler(with_mean=False)

        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.fit_transform(self.X_test)

        return self.X_train, self.X_test
