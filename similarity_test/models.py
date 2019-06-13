"""This module contains different simnet architectures which will be used
during validation.
"""
# keras imports
from keras.models import Model
from keras.layers import Input, Dense, Dropout


def get_base_model(input_shape, num_bit):
    input = Input(shape=input_shape)
    x = Dense(60, activation='relu')(input)
    x = Dropout(0.2)(x)
    x = Dense(80, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(100, activation='relu')(x)
    x = Dense(num_bit, activation='sigmoid')(x)
    return Model(input, x)
