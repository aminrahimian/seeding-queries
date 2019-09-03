# credit: https://networkx.github.io/documentation/stable/auto_examples/drawing/plot_degree_histogram.html
# credit: https://stackoverflow.com/questions/18261587/python-networkx-remove-nodes-and-edges-with-some-condition

from settings import *

import collections
import matplotlib.pyplot as plt
import math

plt.rcParams.update({'font.size': 22})

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
edge_size = G.size()

print('network id', network_id, 'original')
print('node set size', network_size)
print('edge set size', edge_size)

degree_sequence = sorted([d for n, d in G.degree()], reverse=True)  # degree sequence

print(degree_sequence)

print(degree_sequence[99])


print('avg degree', np.mean(degree_sequence))


#draw graph in inset
# plt.axes([0.4, 0.4, 0.5, 0.5])

remove = [node for node,degree in dict(G.degree()).items() if degree < degree_sequence[99]]

G.remove_nodes_from(remove)

network_size = NX.number_of_nodes(G)
edge_size = G.size()

print('node set size', network_size)
print('edge set size', edge_size)

Gcc = sorted(NX.connected_component_subgraphs(G), key=len, reverse=True)[0]
pos = NX.spring_layout(G)
plt.axis('off')
NX.draw_networkx_nodes(G, pos, node_size=10)
NX.draw_networkx_edges(G, pos, alpha=0.4)
plt.show()
