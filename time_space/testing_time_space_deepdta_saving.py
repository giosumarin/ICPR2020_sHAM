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
from datahelper_noflag import *
from sys import getsizeof
import timeit
import getopt
import sys
from os import listdir
from os.path import isfile, join

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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

def list_of_dense_indexes(model):
    index_denses = []
    i = 2
    for layer in model.layers[2:]:
        if type(layer) is tf.keras.layers.Dense:
            index_denses.append(i)
        i += 1
    return index_denses

def make_model_pre_post_dense(model, index_dense):
    submodel = tf.keras.Model(inputs=model.input,
                            outputs=model.layers[index_dense-1].output)
    return submodel

def list_of_dense_weights_indexes(list_weights):
    indexes_denses_weights = []
    i = 2
    for w in list_weights[2:]:
        if len(w.shape) == 2:
            indexes_denses_weights.append(i)
        i += 1
    return indexes_denses_weights

def make_huffman(input_x, matr):
    bit_words_machine = 64
    int_from_string, matr, d_rev, bit_words_machine, matr.dtype, min_length_encoded = huffman.do_all_for_me(matr, bit_words_machine)
    compressed_layers = [int_from_string, d_rev]
    return compressed_layers
    
    
def make_sparse_huffman_only_data(input_x, matr):
    times = 5

    bit_words_machine = 64

    matr_shape, int_data, d_rev_data, row_index, cum, expected_c, min_length_encoded = sparse_huffman_only_data.do_all_for_me(matr, bit_words_machine)

    compressed_layers = [int_data, d_rev_data, row_index, cum]
    return compressed_layers
    




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
                print("tipo: ", arg)
                type_compr = arg
            elif opt == "-d":
                print("directory: ", arg)
                directory = arg
            elif opt == "-m":
                print("model_file: ", arg)
                model_file=arg
            elif opt == "-q":
                print("probabilistic quantization", arg)
                q = False if arg == "0" else True
            elif opt == "-s":
                if arg == "davis":
                    # data loading
                    dataset_path = '../nets/DeepDTA/DAVIS/davis/'
                    dataset = DataSet( fpath = dataset_path, ### BUNU ARGS DA GUNCELLE
                                          setting_no = 1, ##BUNU ARGS A EKLE
                                          seqlen = 1200,
                                          smilen = 85,
                                          need_shuffle = False )

                    XD, XT, Y = dataset.parse_data(dataset_path, 1)

                    XD = np.asarray(XD)
                    XT = np.asarray(XT)
                    Y = np.asarray(Y)

                    drugcount = XD.shape[0]
                    targetcount = XT.shape[0]

                    test_set, outer_train_sets = dataset.read_sets(dataset_path, 1)

                    flat_list = [item for sublist in outer_train_sets for item in sublist]

                    label_row_inds, label_col_inds = np.where(np.isnan(Y)==False)
                    trrows = label_row_inds[flat_list]
                    trcol = label_col_inds[flat_list]
                    drug, targ, aff = prepare_interaction_pairs(XD, XT,  Y, trrows, trcol)
                    trrows = label_row_inds[test_set]
                    trcol = label_col_inds[test_set]
                    drug_test, targ_test, aff_test = prepare_interaction_pairs(XD, XT,  Y, trrows, trcol)

                    x_train=[np.array(drug), np.array(targ)]
                    y_train=np.array(aff)
                    x_test=[np.array(drug_test), np.array(targ_test)]
                    y_test=np.array(aff_test)

except getopt.GetoptError:
    # Print something useful
    print (string_error)
    sys.exit(2)

model = tf.keras.models.load_model(model_file)
original_acc = model.evaluate(x_test, y_test)

onlyfiles = [f for f in listdir(directory) if isfile(join(directory, f))]

#print(sorted(onlyfiles))


if type_compr == "pruning" or type_compr == "pruningweightsharing":
    for weights in sorted(onlyfiles):

            if weights[-3:] == ".h5":
                lw = pickle.load(open(directory+weights, "rb"))
                model.set_weights(lw)

                #_, pruning_acc = model.evaluate(x_test, y_test)

                if type_compr == "pruning":
                    pr, acc = weights.split("-")
                    pruning_acc = float(acc[:-3])
                    print("{}% --> {}".format(pr, acc))
                else:
                    pr, ws, acc = weights.split("-")
                    pruning_acc = float(acc[:-3])
                    print("{}% & {} --> {}".format(pr, ws, acc))
                    
                lodi = list_of_dense_indexes(model)
                lodwi = list_of_dense_weights_indexes(lw)
                compressed_list = []
                for i in range(len(lodi)):
                    model_pre_dense = make_model_pre_post_dense(model, lodi[i])
                    dense_input = model_pre_dense.predict(x_test)

                    compressed_list.append(make_sparse_huffman_only_data(dense_input, lw[lodwi[i]]))
                    
                to_save = [compressed_list.pop(0) if i in lodwi else lw[i] for i in range(len(lw))]
                with open("sHAM-"+weights, "wb") as file:
                    pickle.dump(to_save, file)
                

                compressed_list = []

                for i in range(len(lodi)):
                    model_pre_dense = make_model_pre_post_dense(model, lodi[i])
                    dense_input = model_pre_dense.predict(x_test)

                    compressed_list.append(make_huffman(dense_input, lw[lodwi[i]]))
                    
                to_save = [compressed_list.pop(0) if i in lodwi else lw[i] for i in range(len(lw))]
            with open("HAM-"+weights, "wb") as file:
                pickle.dump(to_save, file)
