import keras
import tensorflow as tf
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
import numpy as np
import sys
from compressionNN import pruning
import pickle
import os
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
if len(argument_list)==2:
    p = int(argument_list[1])
else:
    p = [int(argument_list[i]) for i in range (1,len(argument_list))]

div = 4

model = tf.keras.models.load_model(MODEL)

train_weights = model.trainable_weights

index = find_index_first_dense(train_weights)

print(index)

pr_model = pruning.Pruning_NN(model=model, perc_prun_for_dense=p, index_first_dense=index, apply_compression_bias=b, div=div)

pre_pruning_train = pr_model.accuracy(x_train, y_train)
pre_pruning_test = pr_model.accuracy(x_test, y_test)

print("Accuracy prepruning test: ", pre_pruning_test)

pr_model.apply_pruning(list_trainable=model.trainable_weights, untrainable_per_layers=2)
pr_model.set_optimizer(tf.keras.optimizers.SGD(lr=0.001, momentum=0.9, nesterov=True))
pr_model.set_loss(tf.keras.losses.CategoricalCrossentropy())


post_pruning_train = pr_model.accuracy(x_train, y_train)
post_pruning_test = pr_model.accuracy(x_test, y_test)

print("Accuracy postpruning train: ", post_pruning_train)
print("Accuracy postpruning test: ", post_pruning_test)


dataset = tf.data.Dataset.from_tensor_slices(
  (tf.cast(x_train, tf.float32),
   tf.cast(y_train,tf.int64)))
dataset = dataset.shuffle(1000).batch(BATCH_SIZE)


pr_model.train_pr(epochs=100, dataset=dataset, X_train=x_train, y_train=y_train, X_test=x_test, y_test=y_test, step_per_epoch = 10000000, patience=0)

TRAIN_RES = ([pre_pruning_train] + [post_pruning_train] + pr_model.acc_train)
TEST_RES = ([pre_pruning_test] + [post_pruning_test] + pr_model.acc_test)

with open("VGG19_MNIST_pruning.txt", "a+") as tex:
    tex.write("{} pruning {} -->\n {}\n , {}\n\n".format(b, p, TRAIN_RES, TEST_RES))

DIR="pruning"
TO_SAVE = "{}-{}".format(p,round(TEST_RES[-1],2))
if not os.path.isdir(DIR):
    os.mkdir(DIR)


pruned_weights = pr_model.get_pruned_weights()

with open(DIR+"/"+TO_SAVE+".h5", "wb") as file:
    pickle.dump(pruned_weights, file)
