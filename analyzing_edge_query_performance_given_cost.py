from sparsified_models import *

from pathlib import Path

import numpy as np

from functools import partial

from Multipool import Multipool

import multiprocessing

import os


VERBOSE = True

CHECK_FOR_EXISTING_PKL_SAMPLES = False

MULTIPROCESS_SEED_SAMPLE = True

MULTIPROCESS_SAMPLE = True

network_id = 'Penn94'

seed_sample_size = 50

sample_size = 500

num_seed_sample_cpus = 3

num_sample_cpus = 8

CAP = 0.9

Ts = [0, 1, 2, 3, 4, 5, 7, 8, 10, 13, 16, 20, 24, 30, 37, 46, 57, 71, 88, 109, 134, 166, 206, 255, 315, 390]

        
def analyze_cost_vs_performance(query_cost_id):
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

    eps = 0.2
    a = 0.95
    eps_prime = 2 * eps * (1 + a * (1 - eps))
    tau = np.log(1 / eps) * network_size / (eps * k)
    rho = 10
    T = Ts[query_cost_id]

    sampled_nodes = pickle.load(open(root_data_address
                                    + 'sampled_nodes/'
                                    + 'fb100_edge_query_sampled_nodes_Penn94.pkl', 'rb'))
    candidate_nodes = pickle.load(open(root_data_address
                                    + 'sampled_nodes/'
                                    + 'fb100_edge_query_candidate_nodes_Penn94.pkl', 'rb'))  

    sparsified_graph_id = 100000
    eval_sparsified_graph_id = 119500
    
    params_original = {
        'network': G,
        'original_network': G,
        'network_id': network_id,
        'size': network_size,
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
        'sampled_nodes' : sampled_nodes,
        'candidate_nodes' : candidate_nodes,
        'sparsified_graph_id' : sparsified_graph_id,
        'eval_sparsified_graph_id' : eval_sparsified_graph_id,
        'graph_id_interval' : max(Ts),
        'memory': memory,
    }

    if model_id == '_vanilla IC_':
        dynamics = IndependentCascadeEdgeQuerySeeding(params_original)
    else:
        print('model_id is not valid')
        exit()

    spread_results = dynamics.evaluate_model(seed_sample_size = seed_sample_size, 
                                             sample_size = sample_size, 
                                             num_seed_sample_cpus = num_seed_sample_cpus, 
                                             MULTIPROCESS_SEED_SAMPLE = MULTIPROCESS_SAMPLE, 
                                             num_sample_cpus = num_sample_cpus, 
                                             MULTIPROCESS_SAMPLE = MULTIPROCESS_SEED_SAMPLE)

    if VERBOSE:
        print('================================================', "\n",
              T, np.mean(spread_results), np.std(spread_results), "\n",
              '================================================')

    if save_computations:
        seeding_model_folder = "/edge_query/" + network_id + "/"
        data_dump_folder = (spreading_pickled_samples_directory_address
                                                + 'k_' + str(k)
                                                + seeding_model_folder)
        os.makedirs(os.path.dirname(data_dump_folder), exist_ok = True)

        pickle.dump(spread_results, open(data_dump_folder
                                              + 'spread_size_samples_'
                                              + network_group + network_id
                                              + '_T_' + str(T)
                                              + model_id + '.pkl', 'wb'))


if __name__ == '__main__':

    assert do_computations, "we should be in do_computations mode"

    if do_multiprocessing:
        with Multipool(processes=number_CPU) as pool:

            pool.map(analyze_cost_vs_performance, query_cost_id_list)

    else:  # no multi-processing
        # do computations for the original networks:

        for query_cost_id in query_cost_id_list:
            analyze_cost_vs_performance(query_cost_id)
