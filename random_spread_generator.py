from settings import *
from pathlib import Path
import numpy as np
from functools import partial
from Multipool import Multipool
import multiprocessing
import os

num_cpus = 4
num_sample_cpus = 7


def spread(node, sparsified_graph_id):
    connected_components = pickle.load(open(root_data_address 
                                        + 'sparsified_graphs/'
                                        + network_id
                                        + '/sparsified_graph_' + str(sparsified_graph_id) 
                                        + '.pkl', 'rb'))

    for component in connected_components:
        if node in component:
            return component

    return None

def get_total_spread(sparsified_graph_id, seeds):
    spread_set = set()

    for seed in seeds:
        spread_set.update(spread(seed, sparsified_graph_id))

    return len(spread_set)

def get_all_spreads(seeds, graph_id_range):
    partial_get_spread = partial(get_total_spread, seeds = seeds)

    with Multipool(processes = num_sample_cpus) as pool:
        spreads = pool.map(partial_get_spread, graph_id_range)

    return spreads

def generate_random_spread_data():
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

    graph_id_range = list(range(119500, 120000))
    partial_get_all_spreads = partial(get_all_spreads, graph_id_range = graph_id_range)

    seeds_list = [np.random.choice(G.nodes(), size = k, replace = True) for i in range(50)]
    with Multipool(processes = num_cpus) as pool:
        spread_collection = pool.map(partial_get_all_spreads, seeds_list)

    if save_computations:
        seeding_model_folder = "/edge_query/" + network_id + "/"
        data_dump_folder = (spreading_pickled_samples_directory_address
                                                + 'k_' + str(k)
                                                + seeding_model_folder)
        os.makedirs(os.path.dirname(data_dump_folder), exist_ok = True)

        pickle.dump(spread_collection, open(data_dump_folder
                                              + 'random_spread_size_samples_'
                                              + network_group + network_id
                                              + model_id + '.pkl', 'wb'))

if __name__ == '__main__':
    assert do_computations, "we should be in do_computations mode"
    generate_random_spread_data()

