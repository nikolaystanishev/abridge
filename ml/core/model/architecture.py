from tensorflow.keras.layers import Bidirectional
from tensorflow.python.keras.initializers.initializers_v2 import GlorotUniform, GlorotNormal
from tensorflow.python.keras.layers import Dense, Dropout, Input, Embedding, Masking, GlobalAveragePooling1D, GRU, \
    Conv1D, LSTM, InputLayer
from tensorflow.python.keras.layers.core import Activation, Flatten
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.models import Sequential

from ml.tensorflow.ext.transformer import TokenAndPositionEmbedding, TransformerBlock


def lstm_classifier_1(input_shape, embedding_matrix=None):
    inputs = Input(name='inputs', shape=[input_shape])
    layer = Embedding(1000, 64, input_length=input_shape)(inputs)
    layer = LSTM(64, activation='relu')(layer)
    layer = Dropout(0.2)(layer)
    layer = Dense(1, activation='softmax')(layer)

    model = Model(inputs=inputs, outputs=layer)
    return model


def lstm_classifier_2(input_shape, embedding_matrix=None):
    inputs = Input(name='inputs', shape=[input_shape])
    layer = Embedding(1000, 64, input_length=input_shape, mask_zero=True)(inputs)
    layer = Masking(mask_value=0.0)(layer)
    layer = LSTM(64, activation='relu')(layer)
    layer = Dropout(0.2)(layer)
    layer = Dense(1, activation='softmax')(layer)

    model = Model(inputs=inputs, outputs=layer)
    return model


def lstm_classifier_3(input_shape, embedding_matrix=None):
    inputs = Input(name='inputs', shape=[input_shape])
    layer = Dense(input_shape, kernel_initializer=GlorotUniform())(inputs)
    layer = Embedding(1000, 64, input_length=input_shape, mask_zero=True)(layer)
    layer = Masking(mask_value=0.0)(layer)
    layer = LSTM(64, activation='relu')(layer)
    layer = Dropout(0.2)(layer)
    layer = Dense(1, activation='softmax')(layer)

    model = Model(inputs=inputs, outputs=layer)
    return model


def lstm_classifier_4(input_shape, embedding_matrix=None):
    inputs = Input(name='inputs', shape=[input_shape])
    layer = Dense(input_shape, kernel_initializer=GlorotUniform())(inputs)
    layer = Embedding(403937, 200, weights=[embedding_matrix], input_length=input_shape, trainable=False,
                      mask_zero=True)(layer)
    layer = Masking(mask_value=0.0)(layer)
    layer = LSTM(64, activation='relu')(layer)
    layer = Dropout(0.2)(layer)
    layer = Dense(1, activation='softmax')(layer)

    model = Model(inputs=inputs, outputs=layer)
    return model


def lstm_classifier_5(input_shape, embedding_matrix):
    model = Sequential()

    model.add(Embedding(403937, 200, weights=[embedding_matrix], input_length=input_shape, trainable=False,
                        mask_zero=True))
    model.add(Masking(mask_value=0.0))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    return model


def transformer_classifier_1(input_shape, embedding_matrix=None):
    embed_dim = 32  # Embedding size for each token
    num_heads = 2  # Number of attention heads
    ff_dim = 32  # Hidden layer size in feed forward network inside transformer

    inputs = Input(name='inputs', shape=[input_shape])
    layer = TokenAndPositionEmbedding(input_shape, 1000, embed_dim)(inputs)
    layer = TransformerBlock(embed_dim, num_heads, ff_dim)(layer)
    layer = GlobalAveragePooling1D()(layer)
    layer = Dropout(0.1)(layer)
    layer = Dense(20, activation="relu")(layer)
    layer = Dropout(0.1)(layer)
    outputs = Dense(1, activation="softmax")(layer)

    model = Model(inputs=inputs, outputs=outputs)
    return model


def transformer_classifier_2(input_shape, embedding_matrix=None):
    embed_dim = 32  # Embedding size for each token
    num_heads = 2  # Number of attention heads
    ff_dim = 32  # Hidden layer size in feed forward network inside transformer

    inputs = Input(name='inputs', shape=[input_shape])
    layer = TokenAndPositionEmbedding(input_shape, 1000, embed_dim)(inputs)
    layer = Masking(mask_value=0.0)(layer)
    layer = TransformerBlock(embed_dim, num_heads, ff_dim)(layer)
    layer = GlobalAveragePooling1D()(layer)
    layer = Dropout(0.1)(layer)
    layer = Dense(20, activation="relu")(layer)
    layer = Dropout(0.1)(layer)
    outputs = Dense(1, activation="softmax")(layer)

    model = Model(inputs=inputs, outputs=outputs)
    return model


def transformer_classifier_3(input_shape, embedding_matrix=None):
    embed_dim = 32  # Embedding size for each token
    num_heads = 2  # Number of attention heads
    ff_dim = 32  # Hidden layer size in feed forward network inside transformer

    inputs = Input(name='inputs', shape=[input_shape])
    layer = Dense(input_shape, kernel_initializer=GlorotUniform())(inputs)
    layer = TokenAndPositionEmbedding(input_shape, 1000, embed_dim)(layer)
    layer = Masking(mask_value=0.0)(layer)
    layer = TransformerBlock(embed_dim, num_heads, ff_dim)(layer)
    layer = GlobalAveragePooling1D()(layer)
    layer = Dropout(0.1)(layer)
    layer = Dense(20, activation="relu")(layer)
    layer = Dropout(0.1)(layer)
    outputs = Dense(1, activation="softmax")(layer)

    model = Model(inputs=inputs, outputs=outputs)
    return model


def flatten_classifier_1(input_shape, embedding_matrix=None):
    model = Sequential()

    model.add(Input(name='inputs', shape=[input_shape]))
    model.add(Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1], weights=[embedding_matrix],
                        input_length=input_shape, trainable=False,
                        mask_zero=True))
    model.add(Flatten())
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    return model


def lstm_classifier_6(input_shape, embedding_matrix=None):
    model = Sequential()

    model.add(Input(name='inputs', shape=[input_shape]))
    model.add(Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1], weights=[embedding_matrix],
                        input_length=input_shape, trainable=False,
                        mask_zero=True))
    model.add(Masking(mask_value=0.0))
    model.add(LSTM(64))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    return model


def bi_lstm_classifier_1(input_shape, embedding_matrix):
    from keras.layers import Dense, Dropout, Input, Embedding, Masking, LSTM, Activation
    from keras.models import Sequential
    model = Sequential()

    model.add(Input(name='inputs', shape=[input_shape]))
    model.add(Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1], weights=[embedding_matrix],
                        input_length=input_shape, trainable=False,
                        mask_zero=True))
    model.add(Masking(mask_value=0.0))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    return model


def gru_classifier_1(input_shape, embedding_matrix):
    model = Sequential()

    model.add(Input(name='inputs', shape=[input_shape]))
    model.add(Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1], weights=[embedding_matrix],
                        input_length=input_shape, trainable=False,
                        mask_zero=True))
    model.add(GRU(64))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    return model


def cnn_classifier_1(input_shape, embedding_matrix):
    model = Sequential()

    model.add(Input(name='inputs', shape=[input_shape]))
    model.add(Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1], weights=[embedding_matrix],
                        input_length=input_shape, trainable=False,
                        mask_zero=True))
    model.add(Dropout(0.4))
    model.add(Conv1D(600, 3, padding='valid', activation='relu', strides=1))
    model.add(Conv1D(300, 3, padding='valid', activation='relu', strides=1))
    model.add(Conv1D(150, 3, padding='valid', activation='relu', strides=1))
    model.add(Conv1D(75, 3, padding='valid', activation='relu', strides=1))
    model.add(Flatten())
    model.add(Dense(600))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    return model


def transformer_classifier_4(input_shape, embedding_matrix=None):
    embed_dim = 64  # Embedding size for each token
    num_heads = 2  # Number of attention heads
    ff_dim = 64  # Hidden layer size in feed forward network inside transformer

    model = Sequential()

    model.add(Input(name='inputs', shape=[input_shape]))
    model.add(TokenAndPositionEmbedding(input_shape, 1000, embed_dim))
    model.add(Masking(mask_value=0.0))
    model.add(TransformerBlock(embed_dim, num_heads, ff_dim))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.1))
    model.add(Dense(20, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    return model


def transformer_classifier_5(input_shape, embedding_matrix=None):
    embed_dim = 200  # Embedding size for each token
    num_heads = 2  # Number of attention heads
    ff_dim = 64  # Hidden layer size in feed forward network inside transformer

    model = Sequential()

    model.add(Input(name='inputs', shape=[input_shape]))
    model.add(TokenAndPositionEmbedding(input_shape, 1000, embed_dim, embedding_matrix))
    model.add(Masking(mask_value=0.0))
    model.add(TransformerBlock(embed_dim, num_heads, ff_dim))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.1))
    model.add(Dense(20, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    return model


def lstm_classifier_7(input_shape, embedding_matrix):
    model = Sequential()

    model.add(Input(name='inputs', shape=[input_shape]))
    model.add(Dense(input_shape, kernel_initializer=GlorotUniform()))
    model.add(Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1], weights=[embedding_matrix],
                        input_length=input_shape, trainable=False,
                        mask_zero=True))
    model.add(Masking(mask_value=0.0))
    model.add(LSTM(64))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    return model


def lstm_classifier_8(input_shape, embedding_matrix):
    model = Sequential()

    model.add(Input(name='inputs', shape=[input_shape]))
    model.add(Dense(input_shape, kernel_initializer=GlorotNormal()))
    model.add(Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1], weights=[embedding_matrix],
                        input_length=input_shape, trainable=False,
                        mask_zero=True))
    model.add(Masking(mask_value=0.0))
    model.add(LSTM(64))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    return model


def lstm_classifier_9(input_shape, embedding_matrix):
    model = Sequential()

    model.add(Input(name='inputs', shape=[input_shape]))
    model.add(Dense(input_shape, kernel_initializer=GlorotNormal()))
    model.add(Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1], weights=[embedding_matrix],
                        input_length=input_shape, trainable=False,
                        mask_zero=True))
    model.add(Masking(mask_value=0.0))
    model.add(LSTM(64, kernel_regularizer='l1_l2'))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    return model


def lstm_classifier_10(input_shape, embedding_matrix=None):
    model = Sequential()

    model.add(InputLayer(name='inputs', input_shape=[input_shape]))
    model.add(Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1], weights=[embedding_matrix],
                        input_length=input_shape, trainable=False,
                        mask_zero=True))
    model.add(Masking(mask_value=0.0))
    model.add(LSTM(64))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    return model


def bi_lstm_classifier_2(input_shape, embedding_matrix):
    from keras.layers import Dense, Dropout, InputLayer, Embedding, Masking, LSTM, Activation
    from keras.models import Sequential
    model = Sequential()

    model.add(InputLayer(name='inputs', input_shape=[input_shape]))
    model.add(Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1], weights=[embedding_matrix],
                        input_length=input_shape, trainable=False,
                        mask_zero=True))
    model.add(Masking(mask_value=0.0))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    return model


def lstm_classifier_11(input_shape, embedding_matrix=None):
    model = Sequential()

    model.add(InputLayer(name='inputs', input_shape=[input_shape]))
    model.add(Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1], weights=[embedding_matrix],
                        input_length=input_shape, trainable=False,
                        mask_zero=True))
    model.add(Masking(mask_value=0.0))
    model.add(LSTM(64))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    return model


def lstm_classifier_12(input_shape, embedding_matrix=None):
    model = Sequential()

    model.add(InputLayer(name='inputs', input_shape=[input_shape]))
    model.add(Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1], weights=[embedding_matrix],
                        input_length=input_shape, trainable=False,
                        mask_zero=True))
    model.add(Masking(mask_value=0.0))
    model.add(LSTM(64, activation='relu', return_sequences=True))
    model.add(LSTM(64, activation='relu', return_sequences=True))
    model.add(LSTM(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    return model


def transformer_classifier_6(input_shape, embedding_matrix=None):
    embed_dim = 200  # Embedding size for each token
    num_heads = 2  # Number of attention heads
    ff_dim = 64  # Hidden layer size in feed forward network inside transformer

    model = Sequential()

    model.add(Input(name='inputs', shape=[input_shape]))
    model.add(TokenAndPositionEmbedding(input_shape, embedding_matrix.shape[0], embed_dim, embedding_matrix))
    model.add(Masking(mask_value=0.0))
    model.add(TransformerBlock(embed_dim, num_heads, ff_dim))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.1))
    model.add(Dense(20, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    return model


def bi_lstm_classifier_3(input_shape, embedding_matrix):
    from keras.layers import Dense, Dropout, InputLayer, Embedding, Masking, LSTM, Activation
    from keras.models import Sequential
    model = Sequential()

    model.add(InputLayer(name='inputs', input_shape=[input_shape]))
    model.add(Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1], weights=[embedding_matrix],
                        input_length=input_shape, trainable=False,
                        mask_zero=True))
    model.add(Masking(mask_value=0.0))
    model.add(Bidirectional(LSTM(64, activation='leaky_relu', kernel_regularizer='l1_l2')))
    model.add(Dropout(0.6))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    return model


def bi_lstm_classifier_4(input_shape, embedding_matrix):
    from keras.layers import Dense, Dropout, InputLayer, Embedding, Masking, LSTM, Activation
    from keras.models import Sequential
    model = Sequential()

    model.add(InputLayer(name='inputs', input_shape=[input_shape]))
    model.add(Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1], weights=[embedding_matrix],
                        input_length=input_shape, trainable=False,
                        mask_zero=True))
    model.add(Masking(mask_value=0.0))
    model.add(Bidirectional(LSTM(64, activation='relu', kernel_regularizer='l2')))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    return model


architectures = {
    'lstm-classifier-1': lstm_classifier_1,
    'lstm-classifier-2': lstm_classifier_2,
    'lstm-classifier-3': lstm_classifier_3,
    'lstm-classifier-4': lstm_classifier_4,
    'lstm-classifier-5': lstm_classifier_5,
    'transformer-classifier-1': transformer_classifier_1,
    'transformer-classifier-2': transformer_classifier_2,
    'transformer-classifier-3': transformer_classifier_3,
    'flatten-classifier-1': flatten_classifier_1,
    'lstm-classifier-6': lstm_classifier_6,
    'bi-lstm-classifier-1': bi_lstm_classifier_1,
    'gru-classifier-1': gru_classifier_1,
    'cnn-classifier-1': cnn_classifier_1,
    'transformer-classifier-4': transformer_classifier_4,
    'transformer-classifier-5': transformer_classifier_5,
    'lstm-classifier-7': lstm_classifier_7,
    'lstm-classifier-8': lstm_classifier_8,
    'lstm-classifier-9': lstm_classifier_9,
    'lstm-classifier-10': lstm_classifier_10,
    'bi-lstm-classifier-2': bi_lstm_classifier_2,
    'lstm-classifier-11': lstm_classifier_11,
    'lstm-classifier-12': lstm_classifier_12,
    'transformer-classifier-6': transformer_classifier_6,
    'bi-lstm-classifier-3': bi_lstm_classifier_3,
    'bi-lstm-classifier-4': bi_lstm_classifier_4,
}


class ArchitectureFactory:

    @staticmethod
    def create(architecture, input_shape, embedding):
        return architectures[architecture](input_shape, embedding)
