from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Embedding


def lstm_classifier_1(input_shape):
    inputs = Input(name='inputs', shape=[input_shape])
    layer = Embedding(1000, 64, input_length=input_shape)(inputs)
    layer = LSTM(64, activation='relu')(layer)
    layer = Dropout(0.2)(layer)
    layer = Dense(1, activation='softmax')(layer)
    model = Model(inputs=inputs, outputs=layer)
    return model


architectures = {
    'lstm-classifier-1': lstm_classifier_1
}


class ArchitectureFactory:

    @staticmethod
    def create(architecture, input_shape):
        return architectures[architecture](input_shape)
