from settings import *
from pathlib import Path
import numpy as np
from functools import partial
from Multipool import Multipool
import multiprocessing
import os


seed_sample_size = 50
MULTIPROCESS = True
num_cpus = 28
query_costs = [0, 4, 8, 12, 16, 20, 24, 32, 44, 60, 80, 104, 132, 164, 200]


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

def get_costs(id_index, G, graph_ids, sampled_nodes, k, rho):
    nodes = set()
    edges = set()

    total_nodes = 0
    total_edges = 0

    for i in range(k):
        infected_nodes = sampled_nodes[i][:int(rho)]
        sparsified_graph_id = graph_ids[id_index]
        
        for node in infected_nodes:
            spread_set = spread(node, sparsified_graph_id)
            nodes.update(spread_set)
            total_nodes += len(spread_set)

            edge_set = set(G.subgraph(list(spread_set)).edges())
            edges.update(edge_set)
            total_edges += len(edge_set)

            sparsified_graph_id += 1

    return (len(edges), len(nodes), total_edges, total_nodes)

def generate_cost_file(query_cost_index):
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

    rho = query_costs[query_cost_index] / k
    sampled_nodes = pickle.load(open(root_data_address
                                + 'sampled_nodes/'
                                + 'fb100_spread_query_sampled_nodes_Penn94.pkl', 'rb'))
    first_sparsified_graph_id = 100000
    graph_id_interval = max(query_costs)
    graph_ids = [first_sparsified_graph_id + i * graph_id_interval for i in range(seed_sample_size)]

    costs = []
    if MULTIPROCESS:
        partial_get_costs = partial(get_costs, 
                                    G = G,
                                    graph_ids = graph_ids,
                                    sampled_nodes = sampled_nodes,
                                    k = k,
                                    rho = rho)
        with Multipool(processes = num_cpus) as pool:
            costs = pool.map(partial_get_costs, list(range(seed_sample_size)))
    else:
        for id_index in range(seed_sample_size):
            costs.append(get_costs(id_index, G, graph_ids, sampled_nodes, k, rho))

    if save_computations:
        seeding_model_folder = "/spread_query/" + network_id + "/"
        data_dump_folder = (spreading_pickled_samples_directory_address
                                                + 'k_' + str(k)
                                                + seeding_model_folder)
        os.makedirs(os.path.dirname(data_dump_folder), exist_ok = True)

        pickle.dump(costs, open(data_dump_folder + 'cost_samples_'
                                                + network_group + network_id
                                                + '_T_' + str(T)
                                                + model_id + '.pkl', 'wb'))

if __name__ == '__main__':
    assert do_computations, "we should be in do_computations mode"

    if do_multiprocessing:
        with Multipool(processes=number_CPU) as pool:
            pool.map(generate_cost_file, query_cost_id_list)

    else:  # no multi-processing
        # do computations for the original networks:
        for query_cost_id in query_cost_id_list:
            generate_cost_file(query_cost_id)