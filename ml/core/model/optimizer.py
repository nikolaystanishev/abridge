from tensorflow.keras.optimizers import Adam, RMSprop


optimizers = {
    'adam': Adam,
    'rmsprop': RMSprop
}


class Optimizer:

    def __init__(self, optimizer, learning_rate, decay):
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.decay = decay

    def get(self):
        return OptimizerFactory.create(self.optimizer, self.learning_rate, self.decay)


class OptimizerFactory:

    @staticmethod
    def create(optimizer, learning_rate=None, decay=None):
        if learning_rate is None and decay is None:
            return optimizers[optimizer]()
        elif learning_rate is not None and decay is None:
            return optimizers[optimizer](learning_rate=learning_rate)
        elif learning_rate is None and decay is not None:
            return optimizers[optimizer](decay=decay)
        else:
            return optimizers[optimizer](learning_rate=learning_rate, decay=decay)
