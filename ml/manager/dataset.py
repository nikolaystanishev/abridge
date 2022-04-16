import os

import numpy as np


current_file_path = os.path.dirname(__file__)


datasets = {
    'sentiment140': "../datasets/sentiment140/processed/"
}


class DatasetFactory:

    @staticmethod
    def load(dataset):
        path = os.path.join(current_file_path, datasets[dataset])

        X_train = np.load(path + 'X_train.npy')[:256]
        Y_train = np.load(path + 'Y_train.npy')[:256]
        X_test = np.load(path + 'X_test.npy')[:256]
        Y_test = np.load(path + 'Y_test.npy')[:256]

        return X_train, Y_train, X_test, Y_test
