from models import *

from pathlib import Path

import numpy as np

from functools import partial

import multiprocessing

from Multipool import Multipool

import os


network_id = 'Penn94'

VERBOSE = True

CHECK_FOR_EXISTING_PKL_SAMPLES = False

MULTIPROCESS_SAMPLE = False

sample_size = 100

num_sample_cpus = 20

CAP = 0.9

        
def analyze_cost_vs_performance(query_cost):
    #  load in the network and extract preliminary data
    fh = open(edgelist_directory_address + network_group + network_id + '.txt', 'rb')
    G = NX.read_edgelist(fh, delimiter=DELIMITER)
    #  get the largest connected component:
    if not NX.is_connected(G):
        G = max(NX.connected_component_subgraphs(G), key=len)
        print('largest connected component extracted with size ', len(G.nodes()))
    #  remove self loops:
    if len(list(G.selfloop_edges())) > 0:
        print(
            'warning the graph has ' + str(len(list(G.selfloop_edges()))) + ' self-loops that will be removed')
        print('number of edges before self loop removal: ', G.size())
        G.remove_edges_from(G.selfloop_edges())
        print('number of edges before self loop removal: ', G.size())
    network_size = NX.number_of_nodes(G)

    print('network id', network_id, 'original')
    print('network size', network_size)

    if CHECK_FOR_EXISTING_PKL_SAMPLES:
        path = Path(spreading_pickled_samples_directory_address + 'infection_size_original_'
                    + network_group + network_id
                    + model_id + '.pkl')
        if path.is_file():
            print('infection_size_original_'
                  + network_group + network_id
                  + model_id + ' already exists')
            return

    # Running seeding and spreading simulations
    eps = 0.2
    a = 0.95
    eps_prime = 2 * eps * (1 + a * (1 - eps))
    T = int (3 * (delta + np.log(2)) * (k+1) * np.log(network_size) / (eps * eps))
    tau = 0.9 * network_size
    rho = query_cost / k

    params_original = {
        'network': G,
        'original_network': G,
        'size': network_size,
        'add_edges': False,
        'k': k,
        'delta': delta,
        'alpha': alpha,
        'beta': beta,
        'gamma': gamma,
        'eps' : eps,
        'eps_prime' : eps_prime,
        'rho' : rho,
        'T' : T,
        'tau' : tau,
        'memory': memory,
        'rewire': False,
        'rewiring_mode': 'random_random',
        'num_edges_for_random_random_rewiring': None,
    }

    if model_id == '_vanilla IC_':
        dynamics = IndependentCascadeSpreadQuerySeeding(params_original)
    else:
        print('model_id is not valid')
        exit()

    spread_size_sample = dynamics.get_cost_vs_performance(cap = CAP, 
                                                          sample_size = sample_size,
                                                          multiprocess = MULTIPROCESS_SAMPLE,
                                                          num_sample_cpus = num_sample_cpus)

    if VERBOSE:
        print('================================================', "\n",
              'spread size sample: ', spread_size_sample, "\n", 
              '================================================')

    if save_computations:
        seeding_model_folder = "/spread_query/Penn94/"
        data_dump_folder = (spreading_pickled_samples_directory_address
                                                + 'k_' + str(k)
                                                + seeding_model_folder)
        os.makedirs(os.path.dirname(data_dump_folder), exist_ok = True)

        pickle.dump(spread_size_sample, open(data_dump_folder
                                              + 'spread_size_samples_'
                                              + network_group + network_id
                                              + '_query_cost_' + str(query_cost)
                                              + model_id + '.pkl', 'wb'))

if __name__ == '__main__':

    assert do_computations, "we should be in do_computations mode"

    if do_multiprocessing:
        with Multipool(processes=number_CPU) as pool:

            pool.map(analyze_cost_vs_performance, query_cost_list)

    else:  # no multi-processing
        # do computations for the original networks:

        for query_cost in query_cost_list:
            analyze_cost_vs_performance(query_cost)
