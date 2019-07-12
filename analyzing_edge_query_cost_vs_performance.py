from models import *

from pathlib import Path

import numpy as np

from functools import partial

from multipool import Multipool

import multiprocessing

import os


VERBOSE = True

CHECK_FOR_EXISTING_PKL_SAMPLES = False

MULTIPROCESS_DATASET = True

size_of_dataset = 10

num_cpus = 10

CAP = 0.9


def analyze_performance_for_given_rho(rho, G, network_size):
    print(rho)

    eps = 0.2
    a = 0.95
    eps_prime = 2 * eps * (1 + a * (1 - eps))
    T = int (3 * (delta + np.log(2)) * (k+1) * np.log(network_size) / (eps * eps))
    tau = np.log(1 / eps) * network_size / (eps * k)

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
        dynamics = IndependentCascadeEdgeQuerySeeding(params_original)
    else:
        print('model_id is not valid')
        exit()

    avg_spread_size_sample, std_spread_size_sample, num_of_failed_spread, \
        avg_node_discovery_cost_sample, std_node_discovery_cost_sample, \
        avg_edge_discovery_cost_sample, std_edge_discovery_cost_sample \
            = dynamics.get_cost_vs_performance(cap = CAP, sample_size = 1)

    return ((avg_spread_size_sample, std_spread_size_sample, num_of_failed_spread),
            (avg_node_discovery_cost_sample, std_node_discovery_cost_sample),
            (avg_edge_discovery_cost_sample, std_edge_discovery_cost_sample))

        
def analyze_cost_vs_performance(network_id):
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
    rhos = [0.02*j for j in range(1, size_of_dataset + 1)]
    spread_size_samples = []
    node_discovery_cost_samples = []
    edge_discovery_cost_samples = []

    if MULTIPROCESS_DATASET:
        analyze_performance_partial = partial(analyze_performance_for_given_rho,
                                              G = G,
                                              network_size = network_size)
        with Multipool(processes = num_cpus) as pool:
            spread_results = pool.map(analyze_performance_partial, rhos)
        spread_size_samples = [result[0] for result in spread_results]
        node_discovery_cost_samples = [result[1] for result in spread_results]
        edge_discovery_cost_samples = [result[2] for result in spread_results]
    else:
        for rho in range(rhos):
            spread_result = analyze_performance_for_given_rho(rho, G, network_size)
            spread_size_samples.append(spread_result[0])
            node_discovery_cost_samples.append(spread_result[1])
            edge_discovery_cost_samples.append(spread_result[2])

    if VERBOSE:
        print('================================================', "\n",
              'spread size samples: ', spread_size_samples, "\n",
              'node discovery query cost samples: ', node_discovery_cost_samples, "\n", 
              'edge discovery query cost samples: ', edge_discovery_cost_samples, "\n",
              '================================================')

    if save_computations:
        seeding_model_folder = "/edge_query/"
        data_dump_folder = (spreading_pickled_samples_directory_address
                                                + 'k_' + str(k)
                                                + seeding_model_folder)
        os.makedirs(os.path.dirname(data_dump_folder), exist_ok = True)

        pickle.dump(spread_size_samples, open(data_dump_folder
                                         + 'spread_size_samples_'
                                         + network_group + network_id
                                         + model_id + '.pkl', 'wb'))

        pickle.dump(node_discovery_cost_samples, open(data_dump_folder
                                                 + 'node_discovery_cost_samples_'
                                                 + network_group + network_id
                                                 + model_id + '.pkl', 'wb'))

        pickle.dump(edge_discovery_cost_samples, open(data_dump_folder
                                                 + 'edge_discovery_cost_samples_'
                                                 + network_group + network_id
                                                 + model_id + '.pkl', 'wb'))


if __name__ == '__main__':

    assert do_computations, "we should be in do_computations mode"

    if do_multiprocessing:
        with Multipool(processes=number_CPU) as pool:

            pool.map(analyze_cost_vs_performance, network_id_list)

    else:  # no multi-processing
        # do computations for the original networks:

        for network_id in network_id_list:
            analyze_cost_vs_performance(network_id)