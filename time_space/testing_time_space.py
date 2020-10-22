import keras
from heapq import heappush, heappop, heapify
import numpy as np
from joblib import Parallel, delayed
from scipy.sparse import csc_matrix
from numba import njit, prange, cuda
import tensorflow as tf
from keras.datasets import mnist, cifar10
from keras.utils import np_utils
from compressionNN import huffman
from compressionNN import sparse_huffman
from compressionNN import sparse_huffman_only_data
import pickle
import os
from math import floor
from sys import getsizeof
import timeit
import getopt
import sys
from os import listdir
from os.path import isfile, join

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def dict_space(dict_):
    space_byte = 0
    for key in dict_:
        space_byte += getsizeof(key)-getsizeof("") + 1 #byte end string
        space_byte += 4 #byte for float32
        space_byte += 8 #byte for structure dict
    return space_byte

def dense_space(npmatr2d):
    if npmatr2d.dtype == np.float64:
        byte = 64/8
    elif npmatr2d.dtype == np.float32:
        byte = 32/8
    return npmatr2d.shape[0]*npmatr2d.shape[1]*byte

def list_of_dense_indexes(model):
    index_denses = []
    i = 0
    for layer in model.layers:
        if type(layer) is tf.keras.layers.Dense:
            index_denses.append(i)
        i += 1
    return index_denses

def make_model_pre_post_dense(model, index_dense):
    model_post_dense = tf.keras.Sequential()
    model_pre_dense = tf.keras.Sequential()

    for layer in model.layers[:index_dense]:
        model_pre_dense.add(layer)
    for layer in model.layers[:index_dense+1]:
        model_post_dense.add(layer)

    return model_pre_dense, model_post_dense

def list_of_dense_weights_indexes(list_weights):
    indexes_denses_weights = []
    i = 0
    for w in list_weights:
        if len(w.shape) == 2:
            indexes_denses_weights.append(i)
        i += 1
    return indexes_denses_weights

def make_huffman(input_x, matr):
    times = 20
    bit_words_machine = 64

    int_from_string, matr, d_rev, bit_words_machine, matr.dtype, min_length_encoded = huffman.do_all_for_me(matr, bit_words_machine)

    t_huff = timeit.timeit(lambda:huffman.dot_for_col(input_x, int_from_string, matr, d_rev, bit_words_machine, matr.dtype, min_length_encoded), number=times)/times

    t_np = timeit.timeit(lambda:np.dot(input_x, matr), number=times)/times

    space_dense = dense_space(matr)
    space_huffman = dict_space(d_rev) #space dict coded --> weight
    space_huffman += bit_words_machine/4 * (len(int_from_string)) #space for list of int representing the encoded values converted to integers

    return space_dense, space_huffman, t_np, t_huff

def make_sparse_huffman(input_x, matr):
    times = 20

    bit_words_machine = 64

    matr_shape, int_data, int_row_index, int_cum, d_rev_data, d_rev_row_index, d_rev_cum, expected_c, min_length_encoded_d, min_length_encoded_r, min_length_encoded_c = sparse_huffman.do_all_for_me(matr, bit_words_machine)

    t_huff = timeit.timeit(lambda:sparse_huffman.sparsed_encoded_dot(input_x, matr_shape, int_data, int_row_index, int_cum, d_rev_data, d_rev_row_index, d_rev_cum, bit_words_machine, expected_c, "float32", min_length_encoded_d, min_length_encoded_r, min_length_encoded_c), number=times)/times

    t_np = timeit.timeit(lambda:np.dot(input_x, matr), number=times)/times

    space_dense = dense_space(matr)
    space_sparse_huffman = dict_space(d_rev_data) + dict_space(d_rev_row_index) + dict_space(d_rev_cum) #space of 3 dict coded --> elements
    space_sparse_huffman += bit_words_machine/4 * (len(int_data)+len(int_row_index)+len(int_cum)) #space for 3 lists of int representing the encoded values converted to integers

    return space_dense, space_sparse_huffman, t_np, t_huff

def space_for_row_cum(matr, list_):
    len_ = matr.shape[0]
    if len_ < 2**8:
        return 1 * len(list_)
    elif len_ < 2**16:
        return 2 * len(list_)
    elif len_ < 2**32:
        return 4 * len(list_)
    return 8 * len(list_)

def make_sparse_huffman_only_data(input_x, matr):
    times = 20

    bit_words_machine = 64

    matr_shape, int_data, d_rev_data, row_index, cum, expected_c, min_length_encoded = sparse_huffman_only_data.do_all_for_me(matr, bit_words_machine)

    t_huff = timeit.timeit(lambda:sparse_huffman_only_data.sparsed_encoded_dot(input_x, matr_shape, int_data, d_rev_data, row_index, cum, bit_words_machine, expected_c, "float32", min_length_encoded), number=times)/times
    t_np = timeit.timeit(lambda:np.dot(input_x, matr), number=times)/times

    space_dense = dense_space(matr)
    space_sparse_huffman = dict_space(d_rev_data)  #space of 3 dict coded --> elements
    space_sparse_huffman += bit_words_machine/4 * len(int_data) #space for list of int representing the encoded values converted to integers
    space_sparse_huffman += space_for_row_cum(matr, cum) + space_for_row_cum(matr, row_index)

    return space_dense, space_sparse_huffman, t_np, t_huff



# Get the arguments from the command-line except the filename
argv = sys.argv[1:]

try:
    string_error = 'usage: testing_time_space.py -t <type of compression> -d <directory of compressed weights> -m <file original keras model>'
    # Define the getopt parameters
    opts, args = getopt.getopt(argv, 't:d:m:s:q:', ['type', 'directory', 'model', 'dataset', 'quantization'])
    if len(opts) != 5:
      print (string_error)
      # Iterate the options and get the corresponding values
    else:
        #print(opts)
        #opts
        for opt, arg in opts:
            if opt == "-t":
                print("type: ", arg)
                type_compr = arg
            elif opt == "-d":
                print("directory: ", arg)
                directory = arg
            elif opt == "-m":
                print("model_file: ", arg)
                model_file=arg
            elif opt == "-q":
                print("probabilistic quantization", arg)
                q = False if arg == 0 else True
            elif opt == "-s":
                if arg == "mnist":
                    print(arg)
                    # data loading
                    ((x_train, y_train), (x_test, y_test)) = mnist.load_data()
                    x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
                    x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))
                    x_train = x_train.astype("float32") / 255.0
                    x_test = x_test.astype("float32") / 255.0
                    y_train = np_utils.to_categorical(y_train, 10)
                    y_test = np_utils.to_categorical(y_test, 10)
                elif arg == "cifar10_vgg":
                    # data loading
                    num_classes = 10
                    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
                    y_train = keras.utils.to_categorical(y_train, num_classes)
                    y_test = keras.utils.to_categorical(y_test, num_classes)
                    x_train = x_train.astype('float32')
                    x_test = x_test.astype('float32')

                    # data preprocessing
                    x_train[:,:,:,0] = (x_train[:,:,:,0]-123.680)
                    x_train[:,:,:,1] = (x_train[:,:,:,1]-116.779)
                    x_train[:,:,:,2] = (x_train[:,:,:,2]-103.939)
                    x_test[:,:,:,0] = (x_test[:,:,:,0]-123.680)
                    x_test[:,:,:,1] = (x_test[:,:,:,1]-116.779)
                    x_test[:,:,:,2] = (x_test[:,:,:,2]-103.939)

                elif arg == "cifar10_lenet":
                    num_classes = 10
                    # load data
                    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
                    y_train = keras.utils.to_categorical(y_train, num_classes)
                    y_test = keras.utils.to_categorical(y_test, num_classes)
                    x_train = x_train.astype('float32')
                    x_test = x_test.astype('float32')

                    mean = [125.307, 122.95, 113.865]
                    std = [62.9932, 62.0887, 66.7048]

                    for i in range(3):
                        x_train[:,:,:,i] = (x_train[:,:,:,i] - mean[i]) / std[i]
                        x_test[:,:,:,i] = (x_test[:,:,:,i] - mean[i]) / std[i]
except getopt.GetoptError:
    # Print something useful
    print (string_error)
    sys.exit(2)

model = tf.keras.models.load_model(model_file)
_, original_acc = model.evaluate(x_test, y_test)

onlyfiles = [f for f in listdir(directory) if isfile(join(directory, f))]

pruning_l_h = []
ws_l_h = []
diff_acc_h = []
space_h = []
time_h = []
nonzero_h = []

pruning_l_sh = []
ws_l_sh = []
diff_acc_sh = []
space_sh = []
time_sh = []
nonzero_sh = []


if type_compr == "pruning" or type_compr == "pruningweightsharing":
    for weights in sorted(onlyfiles):
        try:
            if weights[-3:] == ".h5":
                lw = pickle.load(open(directory+weights, "rb"))
                model.set_weights(lw)


                if type_compr == "pruning":
                    pr, acc = weights.split("-")
                    pruning_acc = float(acc[:-3])/100.
                    print("{}% --> {}".format(pr, acc))
                else:
                    pr, ws, acc = weights.split("-")
                    pruning_acc = float(acc[:-3])/100.
                    print("{}% & {} --> {}".format(pr, ws, acc))
                    ws_l_h.append(ws)
                    ws_l_sh.append(ws)

                lodi = list_of_dense_indexes(model)
                lodwi = list_of_dense_weights_indexes(lw)

                assert len(lodi) == len(lodwi)

                original_space = 0
                original_time = 0
                huffman_space = 0
                huffman_time = 0
                non_zero = []

                for i in range(len(lodi)):
                    model_pre_dense, model_post_dense = make_model_pre_post_dense(model, lodi[i])
                    dense_input = model_pre_dense.predict(x_test)

                    space_dense, space_sparse_huffman, t_np, t_huff = make_sparse_huffman_only_data(dense_input, lw[lodwi[i]])
                    original_space += space_dense
                    original_time += t_np
                    huffman_space += space_sparse_huffman
                    huffman_time += t_huff
                    if type_compr == "pruningweightsharing":
                        unique = np.unique(lw[lodwi[i]])
                        non_zero.append(len(unique))
                pruning_l_sh.append(pr)
                diff_acc_sh.append(round(pruning_acc-original_acc, 4))
                space_sh.append(round(huffman_space/original_space,3))
                time_sh.append(floor(huffman_time/original_time))
                if type_compr == "pruningweightsharing":
                    nonzero_sh.append(non_zero)

                print("{}% {} acc1, space {}, time {} ".format(pruning_l_sh[-1], diff_acc_sh[-1], space_sh[-1], time_sh[-1]))

                original_space = 0
                original_time = 0
                huffman_space = 0
                huffman_time = 0
                non_zero = []

                for i in range(len(lodi)):
                    model_pre_dense, model_post_dense = make_model_pre_post_dense(model, lodi[i])
                    dense_input = model_pre_dense.predict(x_test)

                    space_dense, space_huffman, t_np, t_huff = make_huffman(dense_input, lw[lodwi[i]])
                    original_space += space_dense
                    original_time += t_np
                    huffman_space += space_huffman
                    huffman_time += t_huff
                    if type_compr == "pruningweightsharing":
                        unique = np.unique(lw[lodwi[i]])
                        non_zero.append(len(unique))

                pruning_l_h.append(pr)
                diff_acc_h.append(round(pruning_acc-original_acc, 4))
                space_h.append(round(huffman_space/original_space,3))
                time_h.append(floor(huffman_time/original_time))
                if type_compr == "pruningweightsharing":
                    nonzero_h.append(non_zero)

                print("{}% {} acc1, space {}, time {} ".format(pruning_l_h[-1], diff_acc_h[-1], space_h[-1], time_h[-1]))

        except:
            print("ERROR")
            pass
if type_compr == "pruning":
    with open("results/sparse_huffman_pruning.txt", "a+") as tex:
        tex.write(directory)
        tex.write("\npruning = {}\ndiff_acc = {}\nspace = {}\ntime = {}\n".format(pruning_l_sh, diff_acc_sh, space_sh, time_sh))

    with open("results/huffman_pruning.txt", "a+") as tex:
        tex.write(directory)
        tex.write("\npruning = {}\ndiff_acc = {}\nspace = {}\ntime = {}\n".format(pruning_l_h, diff_acc_h, space_h, time_h))

elif type_compr == "pruningweightsharing":
    file_results = "results/sparse_huffman_pruningws.txt" if q == False else "results/sparse_huffman_pruningpq.txt"
    with open(file_results, "a+") as tex:
        tex.write(directory)
        tex.write("\npruning = {}\nclusters = {}\nunique = {}\ndiff_acc = {}\nspace = {}\ntime = {}\n".format(pruning_l_sh, ws_l_sh, nonzero_sh, diff_acc_sh, space_sh, time_sh))


    file_results = "results/huffman_pruningws.txt" if q == False else "results/huffman_pruningpq.txt"
    with open(file_results, "a+") as tex:
        tex.write(directory)
        tex.write("\npruning = {}\nclusters = {}\nunique = {}\ndiff_acc = {}\nspace = {}\ntime = {}\n".format(pruning_l_h, ws_l_h, nonzero_h, diff_acc_h, space_h, time_h))


if type_compr == "weightsharing":
    for weights in sorted(onlyfiles):
        try:
            if weights[-3:] == ".h5":
                lw = pickle.load(open(directory+weights, "rb"))
                model.set_weights(lw)

                #_, ws_acc = model.evaluate(x_test, y_test)

                ws, acc = weights.split("-")
                ws_acc = float(acc[:-3])/100.
                print("{}% --> {}".format(ws, acc))

                lodi = list_of_dense_indexes(model)
                lodwi = list_of_dense_weights_indexes(lw)

                assert len(lodi) == len(lodwi)

                original_space = 0
                original_time = 0
                huffman_space = 0
                huffman_time = 0
                non_zero = []

                for i in range(len(lodi)):
                    model_pre_dense, model_post_dense = make_model_pre_post_dense(model, lodi[i])
                    dense_input = model_pre_dense.predict(x_test)

                    space_dense, space_sparse_huffman, t_np, t_huff = make_huffman(dense_input, lw[lodwi[i]])
                    original_space += space_dense
                    original_time += t_np
                    huffman_space += space_sparse_huffman
                    huffman_time += t_huff
                    unique = np.unique(lw[lodwi[i]])
                    non_zero.append(len(unique))

                ws_l_h.append(ws)
                diff_acc_h.append(round(ws_acc-original_acc, 4))
                space_h.append(round(huffman_space/original_space,3))
                time_h.append(floor(huffman_time/original_time))
                nonzero_h.append(non_zero)


                print("{} {} acc1, space {}, time {} ".format(ws_l_h[-1], diff_acc_h[-1], space_h[-1], time_h[-1]))
        except:
            print("ERROR")
            pass
    file_results = "results/huffman_ws.txt" if q == False else "results/huffman_pq.txt"
    with open(file_results, "a+") as tex:
        tex.write(directory)
        tex.write("\nclusters = {}\nunique = {}\ndiff_acc = {}\nspace = {}\ntime = {}\n".format(ws_l_h, nonzero_h, diff_acc_h, space_h, time_h))
