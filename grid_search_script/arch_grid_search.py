#!/usr/bin/env python
# coding: utf-8

# In[1]:


from time import time
from pickle import load, dump
from itertools import product, starmap
from collections import namedtuple, defaultdict
import sys
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from numpy.random import seed
seed(0)

from keras.models import Model
from keras.layers import Input, Dense, Dropout, Lambda
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping
from keras import backend as K
from keras.utils.vis_utils import plot_model

from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier

from tensorflow import set_random_seed
set_random_seed(0)


# In[2]:


df = pd.read_csv(sys.argv[1], index_col=0)
df.head()


# In[3]:


# load dict with similar/dissimilar movies
with open(sys.argv[3], 'rb') as f:
    movie_sims = load(f)


# In[4]:


def create_pairs(movie_sims, df, split=0.85):
    """Creates positive/negative pairs for one-shot learning"""
    pairs = []
    labels = []

    for movie in movie_sims:
        # get vector for particular movie
        movie_vec = df.loc[movie].values
        # get vectors of its similar/dissimilar movies
        p_vec_l = [df.loc[movie].values for movie in movie_sims[movie]['pos']]
        n_vec_l =[df.loc[movie].values for movie in movie_sims[movie]['neg']]
        # construct pairs
        for pos, neg in zip(p_vec_l, n_vec_l):
            pairs += [[movie_vec, pos]]
            pairs += [[movie_vec, neg]]
            labels += [0, 1]
    
    # validation split
    split_indice = int(split * len(pairs))

    pairs_train = pairs[:split_indice]
    pairs_test = pairs[split_indice:]

    labels_train = labels[:split_indice]
    labels_test = labels[split_indice:]
    
    return ((np.array(pairs_train), np.array(labels_train)),
            (np.array(pairs_test), np.array(labels_test)))


# In[5]:


print('Creating pairs ...')
(x_train, y_train), (x_test, y_test) = create_pairs(movie_sims, df)
print('Pairs created!')
x_test = x_test.astype('float32')


# In[6]:


# loss function
def margin_loss(y_true, y_pred):
    m = 1
    loss = 0.5*(1-y_true)*y_pred + 0.5*y_true*K.maximum(0.0, m - y_pred)
    return loss

def compute_accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.'''
    pred = y_pred.ravel() > 0.5
    return np.mean(pred == y_true)


# In[7]:


def build_model(input_shape,
               l1_shape,
               l2_shape,
               l3_shape,
               l4_shape,
               d1_rate,
               d2_rate,
               distance):
    def build_base_network(input_shape,
                          l1_shape,
                          l2_shape,
                          l3_shape,
                          l4_shape,
                          d1_rate,
                          d2_rate):
        i = Input(shape=input_shape)
        x = Dense(l1_shape, activation='relu')(i)
        x = Dropout(d1_rate)(x)
        x = Dense(l2_shape, activation='relu')(x)
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
        d =  lambda x: K.abs(x[0] - x[1])
    elif distance == 'l2':
        d = lambda x: K.sqrt(K.square(x[0] - x[1]))
    else:
        raise Exception('bad dist')
    
    dist = Lambda(d, 
                  output_shape=lambda x: x[0])([processed_a, processed_b])
    pred = Dense(1, activation='sigmoid')(dist)
    model = Model(input=[input_a, input_b], outputs=[pred])
    model.compile(loss=[margin_loss], optimizer=RMSprop(), metrics=['accuracy'])
    return model


# In[8]:


# grid search parameters
input_shape = [[x_train.shape[-1]]]
l1_shape = [64, 96, 128]
l2_shape = [96, 160, 224]
l3_shape = [128, 256, 384]
l4_shape = [512]

d1_rate = [0.0, 0.25, 0.5, 0.75]
d2_rate = [0.0, 0.25, 0.5, 0.75]

distance = ['l1', 'l2']


# In[9]:


param_grid = dict(
    input_shape=input_shape,
    l1_shape=l1_shape,
    l2_shape=l2_shape,
    l3_shape=l3_shape,
    l4_shape=l4_shape,
    d1_rate=d1_rate,
    d2_rate=d2_rate,
    distance=distance
)


# In[10]:


# self implemented grid search beware


# In[11]:


def named_product(**items):
    Product = namedtuple('Product', items.keys())
    return starmap(Product, product(*items.values()))


# In[12]:


def named_product(**items):
    Product = namedtuple('Product', items.keys())
    return starmap(Product, product(*items.values()))


# In[ ]:

if os.path.isfile(sys.argv[2]):
    print('loaded')
    with open(sys.argv[2], 'rb') as f:
        results = load(f)
else:
    results = dict()


# In[ ]:

total = len([x for x in named_product(**param_grid)])


for i, configuration in enumerate(named_product(**param_grid)):
    print(f'Completed {i}/{total}')
    conf_dict = dict(configuration._asdict())

    if conf_dict in results.values():
        print('omitting')
        continue
    
    model = build_model(**conf_dict)
    epochs = 20
    rms = RMSprop()
    early_stopping = EarlyStopping(patience=4, restore_best_weights=True)
    model.compile(loss=[margin_loss], optimizer=rms, metrics=['accuracy'])
    history = model.fit([x_train[:, 0], x_train[:, 1]], y_train,
                          batch_size=128,
                          epochs=epochs,
                          validation_data=([x_test[:, 0], x_test[:, 1]], y_test), callbacks=[early_stopping],
                          verbose=0)
    
    y_pred = model.predict([x_test[:, 0], x_test[:, 1]])
    te_acc = compute_accuracy(y_test, y_pred)
    results[(i, te_acc)] = conf_dict
    
    if i % 10 == 0:
        with open(sys.argv[2], 'wb') as f:
            dump(results, f)
    


# In[ ]:


with open(sys.argv[2], 'wb') as f:
    dump(results, f)
    
print('Best configuration: ', results[max(results.keys(), key=lambda x: x[1])])


# In[ ]:




