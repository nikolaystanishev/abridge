from tensorflow.keras.layers import Bidirectional, LSTM
from tensorflow.python.keras.initializers.initializers_v2 import GlorotUniform
from tensorflow.python.keras.layers import Dense, Dropout, Input, Embedding, Masking, GlobalAveragePooling1D
from tensorflow.python.keras.layers.core import Activation
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


architectures = {
    'lstm-classifier-1': lstm_classifier_1,
    'lstm-classifier-2': lstm_classifier_2,
    'lstm-classifier-3': lstm_classifier_3,
    'lstm-classifier-4': lstm_classifier_4,
    'lstm-classifier-5': lstm_classifier_5,
    'transformer-classifier-1': transformer_classifier_1,
    'transformer-classifier-2': transformer_classifier_2,
    'transformer-classifier-3': transformer_classifier_3,
}


class ArchitectureFactory:

    @staticmethod
    def create(architecture, input_shape, embedding):
        return architectures[architecture](input_shape, embedding)
