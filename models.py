# -*- "coding: utf-8" -*-

from keras.layers import Input, Embedding, LSTM, Activation, Dropout, Dense, Flatten
from keras.models import Model
from keras.optimizers import Adam
from keras.metrics import categorical_accuracy


def Net(input_dim, output_dim):

    inputs = Input(shape=(input_dim,))
    x = Dense(512, activation="relu")(inputs)
    x = Dense(256, activation="relu")(x)
    x = Dense(512, activation="relu")(x)
    outputs = Dense(output_dim, activation="sigmoid")(x)

    model = Model(inputs, outputs)
    adam = Adam()
    model.compile(optimizer=adam, loss="binary_crossentropy")
    return model

def Net1(input_dim, output_dim):

    inputs = Input(shape=(input_dim,))
    x = Dense(1024, activation="relu")(inputs)
    x = Dense(512, activation="relu")(x)
    x = Dense(1024, activation="relu")(x)
    outputs = Dense(output_dim, activation="sigmoid")(x)

    model = Model(inputs, outputs)
    adam = Adam()
    model.compile(optimizer=adam, loss="binary_crossentropy")
    return model
