import json
import time

from tensorflow.keras.callbacks import Callback


class TimeHistory(Callback):

    def __init__(self, file_name):
        self.file_name = file_name
        self.times = []

    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

    def on_train_end(self, logs=None):
        self.times.append(sum(self.times))
        with open(self.file_name, 'w') as f:
            json.dump(self.times, f)
