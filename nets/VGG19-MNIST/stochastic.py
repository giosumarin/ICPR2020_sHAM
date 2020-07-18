import keras
import tensorflow as tf
from keras.datasets import mnist
from keras import optimizers
import numpy as np
import sys
import pickle
import os
from compressionNN import stochastic
import keras.backend as K
from keras.utils import np_utils

SEED = 1

from numpy.random import seed
seed(SEED)
tf.random.set_seed(SEED)

exec(open("../GPU.py").read())

def find_index_first_dense(list_weights):
    i = 0
    for w in list_weights:
        if len(w.shape)==2:
            return i
        i += 1

BATCH_SIZE = 128

MODEL ='VGG19MNIST.h5'

# data loading
((x_train, y_train), (x_test, y_test)) = mnist.load_data()
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

full_cmd_arguments = sys.argv

argument_list = full_cmd_arguments[1:]

b = True if int(argument_list[0])==1 else False
l = float(argument_list[1])
c = [int(np.log2(int(argument_list[i]))) for i in range (2,len(argument_list))]
c_print = [int(argument_list[i]) for i in range (2,len(argument_list))]


div = 4

model = tf.keras.models.load_model(MODEL)

train_weights = model.trainable_weights

index = find_index_first_dense(train_weights)

print(index)

ws_model = stochastic.Stochastic_NN(model=model, bits_for_dense_layers=c,index_first_dense=index, apply_compression_bias=b, div=div)

pre_ws_train = ws_model.accuracy(x_train, y_train)
pre_ws_test = ws_model.accuracy(x_test, y_test)

print("load weights in ws model, train acc -->", pre_ws_train)
print("load weights in ws model, test acc -->", pre_ws_test)

ws_model.set_optimizer(tf.keras.optimizers.SGD(lr=l, momentum=0.9, nesterov=True))
ws_model.set_loss(tf.keras.losses.CategoricalCrossentropy())

ws_model.apply_stochastic(list_trainable=model.trainable_weights, untrainable_per_layers=2)

post_ws_train = ws_model.accuracy(x_train, y_train)
post_ws_test = ws_model.accuracy(x_test, y_test)

print("apply ws, train acc -->" , post_ws_train)
print("apply ws, test acc -->" , post_ws_test)

dataset = tf.data.Dataset.from_tensor_slices(
  (tf.cast(x_train, tf.float32),
   tf.cast(y_train,tf.int64)))
dataset = dataset.shuffle(1000).batch(BATCH_SIZE)


ws_model.train_ws(epochs=100, lr=l, dataset=dataset, X_train=x_train, y_train=y_train, X_test=x_test, y_test=y_test, patience=0)

lw = ws_model.model.get_weights()

TRAIN_RES = ([pre_ws_train] + [post_ws_train] + ws_model.acc_train)
TEST_RES = ([pre_ws_test] + [post_ws_test] + ws_model.acc_test)

with open("VGG19_MNIST_stochastic.txt", "a+") as tex:
    tex.write("{} lr {} cluster {} -->\n {}\n , {}\n\n".format(b,l, c_print, TRAIN_RES, TEST_RES))

DIR="stochastic"
TO_SAVE = "{}-{}".format(c_print,round(TEST_RES[-1],2))
if not os.path.isdir(DIR):
    os.mkdir(DIR)


ws_weights = ws_model.model.get_weights()

with open(DIR+"/"+TO_SAVE+".h5", "wb") as file:
    pickle.dump(ws_weights, file)
