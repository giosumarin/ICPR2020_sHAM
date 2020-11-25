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
    times = 5
    bit_words_machine = 64

    int_from_string, matr, d_rev, bit_words_machine, matr.dtype, min_length_encoded = huffman.do_all_for_me(matr, bit_words_machine)

    t_huff = timeit.timeit(lambda:huffman.dot_for_col(input_x, int_from_string, matr, d_rev, bit_words_machine, matr.dtype, min_length_encoded), number=times)/times

    t_np = timeit.timeit(lambda:np.dot(input_x, matr), number=times)/times

    space_dense = dense_space(matr)
    space_huffman = dict_space(d_rev) #space dict coded --> weight
    space_huffman += bit_words_machine/4 * (len(int_from_string)) #space for list of int representing the encoded values converted to integers

    return space_dense, space_huffman, t_np, t_huff

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
    times = 5

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
                q = False if arg == 0 else True
            elif opt == "-s":
                if arg == "kiba":
                    print(arg)
                    # data loading
                    dataset_path = '../nets/DeepDTA/KIBA/kiba/'
                    dataset = DataSet( fpath = dataset_path, ### BUNU ARGS DA GUNCELLE
                                          setting_no = 1, ##BUNU ARGS A EKLE
                                          seqlen = 1000,
                                          smilen = 100,
                                          need_shuffle = False )

                    XD, XT, Y = dataset.parse_data(dataset_path, 0)

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

                elif arg == "davis":
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
                    model_pre_dense = make_model_pre_post_dense(model, lodi[i])
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
                diff_acc_sh.append(round(pruning_acc-original_acc, 5))
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
                    model_pre_dense = make_model_pre_post_dense(model, lodi[i])
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
                diff_acc_h.append(round(pruning_acc-original_acc, 5))
                space_h.append(round(huffman_space/original_space,3))
                time_h.append(floor(huffman_time/original_time))
                if type_compr == "pruningweightsharing":
                    nonzero_h.append(non_zero)

                print("{}% {} acc1, space {}, time {} ".format(pruning_l_h[-1], diff_acc_h[-1], space_h[-1], time_h[-1]))


if type_compr == "pruning":
    with open("results/sparse_huffman_pruning_deep.txt", "a+") as tex:
        tex.write(directory)
        tex.write("\npruning = {}\ndiff_acc = {}\nspace = {}\ntime = {}\n".format(pruning_l_sh, diff_acc_sh, space_sh, time_sh))

    with open("results/huffman_pruning_deep.txt", "a+") as tex:
        tex.write(directory)
        tex.write("\npruning = {}\ndiff_acc = {}\nspace = {}\ntime = {}\n".format(pruning_l_h, diff_acc_h, space_h, time_h))

elif type_compr == "pruningweightsharing":
    file_results = "results/sparse_huffman_pruningws_deep.txt" if q == False else "results/sparse_huffman_pruningpq_deep.txt"
    with open(file_results, "a+") as tex:
        tex.write(directory)
        tex.write("\npruning = {}\nclusters = {}\nunique = {}\ndiff_acc = {}\nspace = {}\ntime = {}\n".format(pruning_l_sh, ws_l_sh, nonzero_sh, diff_acc_sh, space_sh, time_sh))

    file_results = "results/huffman_pruningws_deep.txt" if q == False else "results/huffman_pruningpq_deep.txt"
    with open(file_results, "a+") as tex:
        tex.write(directory)
        tex.write("\npruning = {}\nclusters = {}\nunique = {}\ndiff_acc = {}\nspace = {}\ntime = {}\n".format(pruning_l_h, ws_l_h, nonzero_h, diff_acc_h, space_h, time_h))


if type_compr == "weightsharing":
    for weights in sorted(onlyfiles):
            if weights[-3:] == ".h5":
                lw = pickle.load(open(directory+weights, "rb"))
                model.set_weights(lw)

                #_, ws_acc = model.evaluate(x_test, y_test)

                ws, acc = weights.split("-")
                ws_acc = float(acc[:-3])
                print("{}% --> {}".format(ws, acc))

                lodi = list_of_dense_indexes(model)
                lodwi = list_of_dense_weights_indexes(lw)

                assert len(lodi) == len(lodwi)

                original_space = 0
                original_time = 0
                huffman_space = 0
                huffman_time = 0
                non_zero = []

                print(lodi)
                print(lodwi)

                for i in range(len(lodi)):
                    model_pre_dense = make_model_pre_post_dense(model, lodi[i])
                    dense_input = model_pre_dense.predict(x_test)
                    space_dense, space_sparse_huffman, t_np, t_huff = make_huffman(dense_input, lw[lodwi[i]])
                    original_space += space_dense
                    original_time += t_np
                    huffman_space += space_sparse_huffman
                    huffman_time += t_huff
                    unique = np.unique(lw[lodwi[i]])
                    non_zero.append(len(unique))

                ws_l_h.append(ws)
                diff_acc_h.append(round(ws_acc-original_acc, 5))
                space_h.append(round(huffman_space/original_space,3))
                time_h.append(floor(huffman_time/original_time))
                nonzero_h.append(non_zero)


                print("{} {} acc1, space {}, time {} ".format(ws_l_h[-1], diff_acc_h[-1], space_h[-1], time_h[-1]))
    file_results = "results/huffman_ws_deep.txt" if q == False else "results/huffman_pq_deep.txt"
    with open(file_results, "a+") as tex:
        tex.write(directory)
        tex.write("\nclusters = {}\nunique = {}\ndiff_acc = {}\nspace = {}\ntime = {}\n".format(ws_l_h, nonzero_h, diff_acc_h, space_h, time_h))
