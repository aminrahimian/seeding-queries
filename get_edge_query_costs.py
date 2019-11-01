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
Ts = list(range(1, 16))


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

def get_sample_spreads(sampled_nodes, graph_id, T):
    all_spreads = []
    sparsified_graph_id = graph_id

    for i in range(T):
        nodes_already_counted = set()

        for node in sampled_nodes:
            if node not in nodes_already_counted:
                connected_component = spread(node, sparsified_graph_id)
                all_spreads.append(connected_component)
                nodes_already_counted.update(connected_component)

        sparsified_graph_id += 1

    return all_spreads

def get_costs(graph_id, G, sampled_nodes, T):
    all_spreads = get_sample_spreads(sampled_nodes, graph_id, T)

    edge_cost = 0
    node_cost = 0
    for spread in all_spreads:
        subgraph = G.subgraph(list(spread))
        edge_cost += len(subgraph.edges())
        node_cost += len(subgraph.nodes())

    return (edge_cost, node_cost)


def get_costs_for_given_T(T_id):
    T = Ts[T_id]

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

    first_graph_id = 100000
    rho = 10
    interval = max(Ts)
    costs = []
    sampled_nodes = pickle.load(open(root_data_address
                                + 'sampled_nodes/'
                                + 'fb100_edge_query_sampled_nodes_Penn94.pkl', 'rb'))
    sampled_nodes = sorted(sampled_nodes, key = lambda elt : int(elt))[:int(rho)]


    graph_ids = [first_graph_id + i * interval for i in range(seed_sample_size)]
    if not MULTIPROCESS:
        for graph_id in graph_ids:
            costs.append(get_costs(graph_id, G, sampled_nodes, T))
    else:
        partial_get_costs = partial(get_costs, G = G, sampled_nodes = sampled_nodes, T = T)
        with Multipool(processes = num_cpus) as pool:
            costs = pool.map(partial_get_costs, graph_ids)

    if save_computations:
        seeding_model_folder = "/edge_query/" + network_id + "/"
        data_dump_folder = (spreading_pickled_samples_directory_address
                                                + 'k_' + str(k)
                                                + seeding_model_folder)
        os.makedirs(os.path.dirname(data_dump_folder), exist_ok = True)

        pickle.dump(spread_results, open(data_dump_folder
                                                + 'cost_samples_'
                                                + network_group + network_id
                                                + '_T_' + str(T)
                                                + model_id + '.pkl', 'wb'))


if __name__ == '__main__':

    assert do_computations, "we should be in do_computations mode"

    if do_multiprocessing:
        with Multipool(processes=number_CPU) as pool:

            pool.map(get_costs_for_given_T, query_cost_id_list)

    else:  # no multi-processing
        # do computations for the original networks:

        for query_cost_id in query_cost_id_list:
            get_costs_for_given_T(query_cost_id)
