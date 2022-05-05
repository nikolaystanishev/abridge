import os

from tensorflow.python.keras.models import load_model
from tensorflow_addons.metrics import F1Score

from core.util.singleton import Singleton

current_file_path = os.path.dirname(__file__)


class ClassificationModel:
    __metaclass__ = Singleton

    def __init__(self):
        self.model = load_model(os.path.join(current_file_path, 'model.h5'), custom_objects={'F1Score': self.f1_score})

    @staticmethod
    def f1_score():
        return F1Score(num_classes=1, threshold=0.5)

    def compute(self, data):
        return self.model.predict(data)
