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
# 'cai_edgelist_'
# 'chami_union_edgelist_'
# 'chami_advice_edgelist_'
# 'chami_friendship_edgelist_'

if network_group == 'cai_edgelist_':

    root_data_address = '../sociotechnical_proj/' + 'seeding_queries/'+ 'data/cai-data/'

    DELIMITER = ' '

elif network_group == 'chami_friendship_edgelist_':

    root_data_address =  '../sociotechnical_proj/' + 'seeding_queries/'+ 'data/chami-friendship-data/'

    DELIMITER = ','

elif network_group == 'chami_advice_edgelist_':

    root_data_address = '../sociotechnical_proj/' + 'seeding_queries/'+ 'data/chami-advice-data/'

    DELIMITER = ','

elif network_group == 'chami_union_edgelist_':

    root_data_address = '../sociotechnical_proj/' + 'seeding_queries/'+ 'data/chami-union-data/'

    DELIMITER = ','

elif network_group == 'banerjee_combined_edgelist_':

    root_data_address = '../sociotechnical_proj/' + 'seeding_queries/'+ 'data/banerjee-combined-data/'

    DELIMITER = ' '

elif network_group == 'fb100_edgelist_':

    root_data_address = '../sociotechnical_proj/' + 'seeding_queries/'+ 'data/fb100-data/'

    DELIMITER = ' '

    GENERATE_NET_LIST_FROM_AVAILABLE_SAMPLES = False

    TAKE_SMALLEST_N = True

    if TAKE_SMALLEST_N:
        SMALLEST_N = 1

edgelist_directory_address = root_data_address + 'edgelists/'

pickled_samples_directory_address = root_data_address + 'pickled_samples/'

spreading_pickled_samples_directory_address = pickled_samples_directory_address + 'spreading_pickled_samples/'

try:
    os.makedirs(pickled_samples_directory_address)
except OSError as e:
    pass
#    if e.errno != errno.EEXIST:
#        raise

try:
    os.makedirs(spreading_pickled_samples_directory_address)
except OSError as e:
    pass
#    if e.errno != errno.EEXIST:
#        raise

#  different spreading models:
model_id = '_vanilla IC_'

if model_id == '_vanilla IC_':
    MODEL = '_vanilla IC_'
    memory = 0
    alpha = 0.0
    gamma = 1.0
    delta = 0.0
    beta = 0.01
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

network_id_list = ['Penn94']

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

# if 'SLURM_ARRAY_TASK_ID' in os.environ:
#     print('SLURM_ARRAY_TASK_ID: ' + str(os.environ['SLURM_ARRAY_TASK_ID']))
#     JOB_NET_ID = int(os.environ['SLURM_ARRAY_TASK_ID']) - 1
#     NET_ID = network_id_list[JOB_NET_ID]
#     network_id_list = [NET_ID]
#     print('SLURM_ARRAY_TASK_ID: ' + NET_ID)

query_cost_id_list = []

if 'SLURM_ARRAY_TASK_ID' in os.environ:
    print('SLURM_ARRAY_TASK_ID: ' + str(os.environ['SLURM_ARRAY_TASK_ID']))
    QUERY_COST_ID = int(os.environ['SLURM_ARRAY_TASK_ID']) - 1
    query_cost_id_list = [QUERY_COST_ID]
    print('QUERY_COST_ID_LIST:', query_cost_id_list)

# batch_id_list = [11]

# if 'SLURM_ARRAY_TASK_ID' in os.environ:
#     print('SLURM_ARRAY_TASK_ID: ' + str(os.environ['SLURM_ARRAY_TASK_ID']))
#     BATCH_ID = int(os.environ['SLURM_ARRAY_TASK_ID']) - 1
#     batch_id_list = [BATCH_ID]
#     print('BATCH_ID_LIST:', BATCH_ID)

# commonly used settings:
#
# for computations:
do_computations = True
do_multiprocessing = False
save_computations = True

#  check that different modes are set consistently

assert (not save_computations) or do_computations, "do_computations should be true to save_computations"

assert (not do_multiprocessing) or do_computations, "do_computations should be true to do_multiprocessing"


if do_multiprocessing:
    import multiprocessing
    from itertools import product
    if 'SLURM_ARRAY_TASK_ID' in os.environ:
        number_CPU = 3
    else:
        number_CPU = 20

