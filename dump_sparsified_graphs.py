from settings import *
from sparsified_models import *
import os
import networkx as NX
from scipy.stats import bernoulli
import heapq
from functools import partial
from Multipool import Multipool


batch_size = 10000
network_id = 'Penn94'
MULTIPROCESS_BATCH = True
num_batch_cpu = 28


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

    f = lambda graph, u, v : beta
    params = {'network' : G,
              'f' : f,
              'beta' : beta}
    contagion_model = IndependentCascade(params)
    
    contagion_model.generate_sparsified_graphs_by_batch(batch_size = batch_size,
                                                        batch_id = batch_id,
                                                        MULTIPROCESS_BATCH = MULTIPROCESS_BATCH,
                                                        num_batch_cpu = num_batch_cpu)


if __name__ == '__main__':
    assert do_computations, "we should be in do_computations mode"

    if do_multiprocessing:
        with Multipool(processes=number_CPU) as pool:

            pool.map(generate_sparsified_graphs_by_batch, batch_id_list)

    else:  # no multi-processing
        # do computations for the original networks:

        for batch_id in batch_id_list:
            generate_sparsified_graphs_by_batch(batch_id)
