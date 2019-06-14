"""This module contains different simnet architectures which will be used
during validation.
"""
# keras imports
from keras.models import Model
from keras.layers import Lambda, Input, Dense, Dropout
from keras.optimizers import RMSprop
from keras import backend as K

from utils import margin_loss


def build_model(input_shape,
                l1_shape,
                l2_shape,
                l3_shape,
                l4_shape,
                d1_rate,
                d2_rate,
                distance):
    """Returns full model and base network."""
    def build_base_network(input_shape,
                           l1_shape,
                           l2_shape,
                           l3_shape,
                           l4_shape,
                           d1_rate,
                           d2_rate):
        """Returns based network."""
        i = Input(shape=input_shape)
        x = Dense(l1_shape, activation='relu')(i)
        if d1_rate:
            x = Dropout(d1_rate)(x)
        x = Dense(l2_shape, activation='relu')(x)
        if d2_rate:
            x = Dropout(d2_rate)(x)
        x = Dense(l3_shape, activation='relu')(x)
        x = Dense(l4_shape, activation='sigmoid')(x)
        return Model(i, x)

    base_network = build_base_network(input_shape,
                                      l1_shape,
                                      l2_shape,
                                      l3_shape,
                                      l4_shape,
                                      d1_rate,
                                      d2_rate)

    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)

    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    if distance == 'l1':
        d = lambda x: K.abs(x[0] - x[1])
    elif distance == 'l2':
        d = lambda x: K.sqrt(K.square(x[0] - x[1]))
    else:
        raise Exception('bad dist')

    dist = Lambda(d,
                  output_shape=lambda x: x[0])([processed_a, processed_b])
    pred = Dense(1, activation='sigmoid')(dist)
    model = Model(input=[input_a, input_b], outputs=[pred])
    model.compile(loss=[margin_loss], optimizer=RMSprop(),
                  metrics=['accuracy'])
    return model, base_network
