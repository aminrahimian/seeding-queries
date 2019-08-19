from settings import *
import os
import networkx as NX
from scipy.stats import bernoulli
import heapq
from functools import partial
from Multipool import Multipool


batch_size = 25000
network_id = 'Penn94'
MULTIPROCESS_BATCH = True
num_batch_cpu = 28


def generate_live_edge_set(graph, f):
    edges = list(graph.edges())
    edge_probabilities = [f(graph, u, v) for (u, v) in edges]
    live_edge_set = set()

    edge_life_index = bernoulli.rvs(edge_probabilities)
    for j in range(len(edges)):
        if edge_life_index[j]:
            live_edge_set.add(edges[j])

    return live_edge_set

def is_live_edge(live_edge_set, edge):
    return edge in live_edge_set or edge[::-1] in live_edge_set

def compute_connected_component_via_bfs(graph, live_edge_set, v):
    visited_nodes = set()
    bfs_queue = {v}
    connected_component = set()

    while bfs_queue:
        node_to_visit = bfs_queue.pop()
        visited_nodes.add(node_to_visit)
        connected_component.add(node_to_visit)

        for u in graph.neighbors(node_to_visit):
            if u not in visited_nodes and is_live_edge(live_edge_set, (node_to_visit, u)):
                bfs_queue.add(u)

    return connected_component

def generate_connected_components(i, graph):
    print("Processing sparsified graph", i)

    f = lambda G, u, v : beta
    live_edge_set = generate_live_edge_set(graph, f)

    nodes = set(graph.nodes())
    connected_components = []
    while len(nodes) != 0:
        candidate_node = nodes.pop()
        connected_component = compute_connected_component_via_bfs(graph, live_edge_set, candidate_node)
        nodes.difference_update(connected_component)
        connected_components.append(connected_component)

    dump_folder_address = root_data_address + 'sparsified_graphs/'
    os.makedirs(os.path.dirname(dump_folder_address), exist_ok = True)

    pickle.dump(connected_components, open(dump_folder_address
                                           + 'sparsified_graph_' + str(i) 
                                           + '.pkl', 'wb'))    
    print("Done!")

def generate_sparsified_graphs_by_batch(batch_id):
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

    print('network id', network_id, 'original')

    sparsified_graphs_id_list = [(batch_size * batch_id + i) for i in range(batch_size)]

    if MULTIPROCESS_BATCH:
        generate_connected_components_partial = partial(generate_connected_components,
                                                        graph = G)
        with Multipool(processes=num_batch_cpu) as pool:
            pool.map(generate_connected_components_partial, sparsified_graphs_id_list)
    else:
        for sparsified_graphs_id in sparsified_graphs_id_list:
            generate_connected_components(sparsified_graphs_id, G)


if __name__ == '__main__':
    assert do_computations, "we should be in do_computations mode"

    if do_multiprocessing:
        with Multipool(processes=number_CPU) as pool:

            pool.map(generate_sparsified_graphs_by_batch, batch_id_list)

    else:  # no multi-processing
        # do computations for the original networks:

        for batch_id in batch_id_list:
            generate_sparsified_graphs_by_batch(batch_id)
