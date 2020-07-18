import tensorflow as tf
from tensorflow import keras

import numpy as np
import random as rn

exec(open("../../GPU.py").read())

import os
os.environ['PYTHONHASHSEED'] = '0'

np.random.seed(1)
rn.seed(1)

SEED = 1

from numpy.random import seed
seed(SEED)
tf.random.set_seed(SEED)

from datahelper_noflag import *
from itertools import product
#from arguments import argparser, logging


from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D, MaxPooling1D
from tensorflow.keras.layers import Conv2D, GRU
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, TimeDistributed, Masking, RepeatVector, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import optimizers, layers


import sys, pickle, os
import math, json, time
import decimal
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from random import shuffle
from copy import deepcopy
from sklearn import preprocessing
from emetrics import get_aupr, get_cindex, get_rm2

def cindex_score(y_true, y_pred):

    g = tf.subtract(tf.expand_dims(y_pred, -1), y_pred)
    g = tf.cast(g == 0.0, tf.float32) * 0.5 + tf.cast(g > 0.0, tf.float32)

    f = tf.subtract(tf.expand_dims(y_true, -1), y_true) > 0.0
    f = tf.linalg.band_part(tf.cast(f, tf.float32), -1, 0)

    g = tf.reduce_sum(tf.multiply(g, f))
    f = tf.reduce_sum(f)

    return tf.where(tf.equal(g, 0), 0.0, g/f)

def prepare_interaction_pairs(XD, XT,  Y, rows, cols):
    drugs = []
    targets = []
    targetscls = []
    affinity=[]

    for pair_ind in range(len(rows)):
        drug = XD[rows[pair_ind]]
        drugs.append(drug)

        target=XT[cols[pair_ind]]
        targets.append(target)

        affinity.append(Y[rows[pair_ind],cols[pair_ind]])

    drug_data = np.stack(drugs)
    target_data = np.stack(targets)

    return drug_data,target_data,  affinity

def build_combined_categorical(NUM_FILTERS, FILTER_LENGTH1, FILTER_LENGTH2, max_smi_len, max_seq_len, charsmiset_size, charseqset_size):

    XDinput = Input(shape=(max_smi_len,), dtype='int32')
    XTinput = Input(shape=(max_seq_len,), dtype='int32')

    encode_smiles = Embedding(input_dim=charsmiset_size+1, output_dim=128, input_length=max_smi_len)(XDinput)
    encode_smiles = Conv1D(filters=NUM_FILTERS, kernel_size=FILTER_LENGTH1,  activation='relu', padding='valid',  strides=1)(encode_smiles)
    encode_smiles = Conv1D(filters=NUM_FILTERS*2, kernel_size=FILTER_LENGTH1,  activation='relu', padding='valid',  strides=1)(encode_smiles)
    encode_smiles = Conv1D(filters=NUM_FILTERS*3, kernel_size=FILTER_LENGTH1,  activation='relu', padding='valid',  strides=1)(encode_smiles)
    encode_smiles = GlobalMaxPooling1D()(encode_smiles)


    encode_protein = Embedding(input_dim=charseqset_size+1, output_dim=128, input_length=max_seq_len)(XTinput)
    encode_protein = Conv1D(filters=NUM_FILTERS, kernel_size=FILTER_LENGTH2,  activation='relu', padding='valid',  strides=1)(encode_protein)
    encode_protein = Conv1D(filters=NUM_FILTERS*2, kernel_size=FILTER_LENGTH2,  activation='relu', padding='valid',  strides=1)(encode_protein)
    encode_protein = Conv1D(filters=NUM_FILTERS*3, kernel_size=FILTER_LENGTH2,  activation='relu', padding='valid',  strides=1)(encode_protein)
    encode_protein = GlobalMaxPooling1D()(encode_protein)


    encode_interaction = keras.layers.concatenate([encode_smiles, encode_protein], axis=-1) #merge.Add()([encode_smiles, encode_protein])

    # Fully connected
    FC1 = Dense(1024, activation='relu')(encode_interaction)
    FC2 = Dropout(0.1)(FC1)
    FC2 = Dense(1024, activation='relu')(FC2)
    FC2 = Dropout(0.1)(FC2)
    FC2 = Dense(512, activation='relu')(FC2)


    # And add a logistic regression on top
    predictions = Dense(1, kernel_initializer='normal')(FC2) #OR no activation, rght now it's between 0-1, do I want this??? activation='sigmoid'

    interactionModel = Model(inputs=[XDinput, XTinput], outputs=[predictions])

    interactionModel.compile(optimizer='adam', loss='mean_squared_error')#, metrics=[cindex_score]) #, metrics=['cindex_score']
    print(interactionModel.summary())
    #plot_model(interactionModel, to_file='figures/build_combined_categorical.png')

    return interactionModel







num_windows = 32
seq_window_lengths = 12
smi_window_lengths = 8
batch_size = 256
num_epoch = 100
max_seq_len = 1000
max_smi_len = 100
dataset_path = 'kiba/'
problem_type = 1
NUM_FILTERS = 32
FILTER_LENGTH1 = seq_window_lengths
FILTER_LENGTH2 = smi_window_lengths


dataset = DataSet( fpath = dataset_path, ### BUNU ARGS DA GUNCELLE
                      setting_no = problem_type, ##BUNU ARGS A EKLE
                      seqlen = max_seq_len,
                      smilen = max_smi_len,
                      need_shuffle = False )

charseqset_size = dataset.charseqset_size
charsmiset_size = dataset.charsmiset_size

XD, XT, Y = dataset.parse_data(dataset_path, 0)

XD = np.asarray(XD)
XT = np.asarray(XT)
Y = np.asarray(Y)

drugcount = XD.shape[0]
print(drugcount)
targetcount = XT.shape[0]
print(targetcount)

test_set, outer_train_sets = dataset.read_sets(dataset_path, problem_type)

flat_list = [item for sublist in outer_train_sets for item in sublist]

label_row_inds, label_col_inds = np.where(np.isnan(Y)==False)
trrows = label_row_inds[flat_list]
trcol = label_col_inds[flat_list]
drug, targ, aff = prepare_interaction_pairs(XD, XT,  Y, trrows, trcol)
trrows = label_row_inds[test_set]
trcol = label_col_inds[test_set]
drug_test, targ_test, aff_test = prepare_interaction_pairs(XD, XT,  Y, trrows, trcol)

batch_size = 256
input_set = tf.data.Dataset.from_tensor_slices((np.array(drug), np.array(targ)))
output_set = tf.data.Dataset.from_tensor_slices((np.array(aff)))
dataset = tf.data.Dataset.zip((input_set, output_set))
dataset = dataset.batch(batch_size)

model = build_combined_categorical(NUM_FILTERS, FILTER_LENGTH1, FILTER_LENGTH2, max_smi_len, max_seq_len, charsmiset_size, charseqset_size)

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)

model.fit(dataset, epochs=num_epoch, validation_data=( ([np.array(drug_test), np.array(targ_test) ]), np.array(aff_test)),  shuffle=False, callbacks=[es] )

model.save("deepDTA_kiba.h5")
