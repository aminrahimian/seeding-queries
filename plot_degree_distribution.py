# credit: https://networkx.github.io/documentation/stable/auto_examples/drawing/plot_degree_histogram.html
# credit: https://stackoverflow.com/questions/44068435/setting-both-axes-logarithmic-in-bar-plot-matploblib/51825144

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

print('network id', network_id, 'original')
print('network size', network_size)
degree_sequence = sorted([d for n, d in G.degree()], reverse=True)  # degree sequence
# print "Degree sequence", degree_sequence
degreeCount = collections.Counter(degree_sequence)
deg, cnt = zip(*degreeCount.items())
print(deg)
print(cnt)
fig, ax = plt.subplots()
x = np.logspace(0, 3.65, num=10)
x = [math.floor(xx) for xx in x]
x[-1] = deg[-1]

plt.bar(deg[:-1], cnt[:-1], width=np.diff(deg), ec="k", align="edge")
plt.title("Degree Histogram")
plt.ylabel("Count")
plt.xlabel("Degree")
ax.set_xscale("log")
ax.set_xticks(x)
ax.set_xticklabels(x)
#draw graph in inset
plt.axes([0.4, 0.4, 0.5, 0.5])
Gcc = sorted(NX.connected_component_subgraphs(G), key=len, reverse=True)[0]
pos = NX.spring_layout(G)
plt.axis('off')
NX.draw_networkx_nodes(G, pos, node_size=0.4)
NX.draw_networkx_edges(G, pos, alpha=0.4)
plt.show()
