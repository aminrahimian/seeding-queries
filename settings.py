# Params and other settings are set here
# Settings are for the generative model as well as the inference engine

# The generative model settings

import random as RD
import numpy as np
import pickle
import os
import errno
import networkx as NX
import re

susceptible = 0.0
infected = 1.0

active = 0.5
inactive = -0.5

SIMPLE = 0
COMPLEX = 1

SENTINEL = object()


def get_n_smallest_key_values(dictionary, n):
    smallest_entries = sorted(
        dictionary.keys(), key=lambda t: dictionary[t], reverse=False)[:n]
    return smallest_entries


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split('(\d+)', text)]


# real world networks simulation settings:
network_group = 'fb100_edgelist_'
# 'banerjee_combined_edgelist_'
#'chami_union_edgelist_'
# 'chami_union_edgelist_'
# 'fb100_edgelist_'
# 'cai_edgelist_'
# 'chami_advice_edgelist_'
# 'banerjee_combined_edgelist_'
# 'cai_edgelist_' #'fb100_edgelist_'
# 'banerjee_combined_edgelist_'
# 'chami_friendship_edgelist_'
# 'chami_advice_edgelist_'
# 'cai_edgelist_'

if network_group == 'cai_edgelist_':

    root_data_address = './data/cai-data/'

    DELIMITER = ' '

elif network_group == 'chami_friendship_edgelist_':

    root_data_address = './data/chami-friendship-data/'

    DELIMITER = ','

elif network_group == 'chami_advice_edgelist_':

    root_data_address = './data/chami-advice-data/'

    DELIMITER = ','

elif network_group == 'chami_union_edgelist_':

    root_data_address = './data/chami-union-data/'

    DELIMITER = ','

elif network_group == 'chami_union_edgelist_':

    root_data_address = './data/chami-union-data/'

    DELIMITER = ','

elif network_group == 'banerjee_combined_edgelist_':

    root_data_address = './data/banerjee-combined-data/'

    DELIMITER = ' '

elif network_group == 'fb100_edgelist_':

    root_data_address = './data/fb100-data/'

    DELIMITER = ' '

    GENERATE_NET_LIST_FROM_AVAILABLE_SAMPLES = False

    TAKE_SMALLEST_N = True

    if TAKE_SMALLEST_N:
        SMALLEST_N = 1

edgelist_directory_address = root_data_address + 'edgelists/'

output_directory_address = root_data_address + 'output/'

pickled_samples_directory_address = root_data_address + 'pickled_samples/'

properties_pickled_samples_directory_address = pickled_samples_directory_address + 'properties_pickled_samples/'

spreading_pickled_samples_directory_address = pickled_samples_directory_address + 'spreading_pickled_samples/'

networks_pickled_samples_directory_address = pickled_samples_directory_address + 'networks_pickled_samples/'

use_separate_address_for_pickled_networks = False  # pickled_networks take a lot of space.
# Some may need to put them else where with a lot of space away from other pickled samples.

separate_address_for_pickled_networks = \
    '/home/rahimian/seeding_queries/data/pickled_networks/'
# '/home/rahimian/seeding_queries/data/pickled_networks/'
#  '/home/amin/Desktop/pickled_networks/'

if use_separate_address_for_pickled_networks:
    networks_pickled_samples_directory_address = \
        separate_address_for_pickled_networks

try:
    os.makedirs(output_directory_address)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

try:
    os.makedirs(pickled_samples_directory_address)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

try:
    os.makedirs(properties_pickled_samples_directory_address)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

try:
    os.makedirs(spreading_pickled_samples_directory_address)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

try:
    os.makedirs(networks_pickled_samples_directory_address)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

#  different spreading models:

model_id = '_vanilla IC_'

if model_id == '_vanilla IC_':
    MODEL = '_vanilla IC_'
    memory = 0
    alpha = 0.0
    gamma = 1.0
    delta = 0.0
    beta = 0.1
    k = 4
else:
    print('model_id is not valid')
    exit()

network_id_list = []

for file in os.listdir(edgelist_directory_address):
    filename = os.path.splitext(file)[0]
    net_id = filename.replace(network_group, '')
    print(net_id)
    network_id_list += [net_id]

network_id_list.sort(key=natural_keys)

print('without checking the availability of samples or taking smaller ones:')

print(network_id_list)

network_id_list = ['MIT8']

try:
    if GENERATE_NET_LIST_FROM_AVAILABLE_SAMPLES == True:
        network_id_list = []
        for file in os.listdir(edgelist_directory_address):
            filename = os.path.splitext(file)[0]
            net_id = filename.replace(network_group, '')
            print(net_id)
            available_sample_file = 'infection_size_samples_' + '10' + '_percent_' \
                                    + 'add_triad_'\
                                    + network_group + net_id + model_id + '.pkl'
            print(available_sample_file)
            print(spreading_pickled_samples_directory_address)
            if available_sample_file in os.listdir(spreading_pickled_samples_directory_address):
                network_id_list += [net_id]
            else:
                print(net_id + ' has no samples available!')

        network_id_list.sort(key=natural_keys)

        print('before taking smallest N:')

        print(network_id_list)

    if TAKE_SMALLEST_N:

        assert SMALLEST_N <= len(network_id_list), "not enough nets in the net_id list"

        net_id_dic = dict.fromkeys(network_id_list)

        for network_id in net_id_dic.keys():

            print('loading' + network_group + network_id)

            #  load in the network and extract preliminary data

            fh = open(edgelist_directory_address + network_group + network_id + '.txt', 'rb')

            G = NX.read_edgelist(fh, delimiter=DELIMITER)

            print('original size ', len(G.nodes()))

            #  get the largest connected component:
            if not NX.is_connected(G):
                G = max(NX.connected_component_subgraphs(G), key=len)
                print('largest connected component extracted with size ', len(G.nodes()))

            network_size = NX.number_of_nodes(G)

            net_id_dic[network_id] = network_size

        print(net_id_dic)

        network_id_list = get_n_smallest_key_values(net_id_dic,SMALLEST_N)

        network_id_list.sort(key=natural_keys)

        print('after taking smallest N')

        print(network_id_list)



except NameError:

    print('could not check for availability of samples or take smaller ones')

# check for SLURM Job Array environmental variable:
if 'SLURM_ARRAY_TASK_ID' in os.environ:
    print('SLURM_ARRAY_TASK_ID: ' + str(os.environ['SLURM_ARRAY_TASK_ID']))
    JOB_NET_ID = int(os.environ['SLURM_ARRAY_TASK_ID']) - 1
    NET_ID = network_id_list[JOB_NET_ID]
    network_id_list = [NET_ID]
    print('SLURM_ARRAY_TASK_ID: ' + NET_ID)

# theory simulations settings:

simulation_type = 'ck_union_ER_vs_size'
#  'c1_c2_interpolation'
#  'ck_union_ER_vs_size'
#  'ck_union_ER_vs_k'
#  'c1_c2_interpolation'
#  'c1_union_ER'
#  'c1_c2_interpolation_SimpleOnlyAlongC1'

root_theory_simulations_address = './data/theory-simulations/'

try:
    os.makedirs(root_theory_simulations_address)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

theory_simulation_output_address = root_theory_simulations_address + simulation_type + '/output/'

theory_simulation_pickle_address = root_theory_simulations_address + simulation_type + '/pickled_samples/'

try:
    os.makedirs(theory_simulation_output_address)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

try:
    os.makedirs(theory_simulation_pickle_address)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

# commonly used settings:
#
# for computations:
do_computations = True
do_multiprocessing = False
save_computations = True
load_computations = False
do_plots = False
save_plots = False
show_plots = False
data_dump = False
simulator_mode = False
#
# for plotting:
# do_computations = False
# do_multiprocessing = False
# save_computations = False
# load_computations = True
# do_plots = True
# save_plots = True
# show_plots = False
# data_dump = False
# simulator_mode = False
#
# # # for data_dump:
# do_computations = False
# do_multiprocessing = False
# save_computations = False
# load_computations = True
# do_plots = False
# save_plots = False
# show_plots = False
# data_dump = True
# simulator_mode = False

# for simulator:
# do_computations = True
# do_multiprocessing = False
# save_computations = False
# load_computations = False
# # simulator uses a different mathplotlib setting for plotting
# do_plots = False
# save_plots = False
# show_plots = False
# data_dump = False
# simulator_mode = True

#  check that different modes are set consistently

assert (not save_computations) or do_computations, "do_computations should be true to save_computations"

assert (not do_multiprocessing) or do_computations, "do_computations should be true to do_multiprocessing"

assert (not (save_plots or show_plots)) or do_plots, "do_plots should be true to save_plots or show_plots"

assert (not do_plots) or save_plots or show_plots, "when do_plots should either save_plots or show_plots"

assert not (load_computations and save_computations), "cannot both save_computations and load_computations"

assert not (load_computations and do_computations), "cannot both do_computations and load_computations"

assert not (show_plots and save_plots), "you can either save plots or show plots not both"

assert not (data_dump and (save_plots or show_plots)), "you cannot do any plots in data_dump mode"

assert not (data_dump and (save_computations or do_computations)), "you cannot do any computations in data_dump mode"

assert (not data_dump) or load_computations, "load_computations should be true to data_dump"

assert not (simulator_mode and do_plots), "simulator_mode and do_plots use different " \
                                          "(conflicting) matplotlib settings, and " \
                                          "cannot be both true"



# import the required packages for different modes:

if data_dump:
    import csv
    import pandas as pd

if do_plots:
    import matplotlib.pyplot as plt
    plt.rc('text', usetex=True)
    plt.rc('font', ** {'family': 'sans-serif', 'sans-serif':['Helvetica']})
    plt.rcParams.update({'font.size': 15})
    from decimal import Decimal
    FOURPLACES = Decimal(10) ** -4
    TWOPLACES = Decimal(10) ** -2
    ERROR_BAR_TYPE = 'std'

if do_multiprocessing:
    import multiprocessing
    from itertools import product
    if 'SLURM_ARRAY_TASK_ID' in os.environ:
        number_CPU = 3
    else:
        number_CPU = 20


def combine(list_of_names,output_name):
    '''Useful for combining pickle files from cluster computations.'''
    original = pickle.load(open('./data/' + list_of_names[0] + '.pkl', 'rb'))
    for ii in range(1, len(list_of_names)):
        additional = pickle.load(open('./data/' + list_of_names[ii] + '.pkl', 'rb'))
        print(original)
        print(additional)
        original = np.concatenate((original,additional),axis=1)
        print(original)
    pickle.dump(original, open('./data/'+output_name+'.pkl', 'wb'))
