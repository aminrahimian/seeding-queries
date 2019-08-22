from settings import *
import gc
import copy
import itertools
import heapq
from scipy.stats import bernoulli
from functools import partial
from Multipool import Multipool


class ContagionModel(object):
    def __init__(self, params):
        self.params = params

    def generate_live_edge_set(self):
        pass

    def is_live_edge(self, live_edge_set, edge):
        pass

    def compute_connected_component_via_bfs(self, live_edge_set, v):
        pass

    def generate_connected_components(self, i):
        pass

    def generate_sparsified_graphs_by_batch(self, batch_size, batch_id, 
                                            MULTIPROCESS_BATCH = True, 
                                            num_batch_cpu = 28):
        pass


class IndependentCascade(ContagionModel):
    def __init__(self, params):
        super(IndependentCascade, self).__init__(params)

    def generate_live_edge_set(self):
        graph = self.params['network']
        f = self.params['f']

        edges = list(graph.edges())
        edge_probabilities = [f(graph, u, v) for (u, v) in edges]
        live_edge_set = set()

        edge_life_index = bernoulli.rvs(edge_probabilities)
        for j in range(len(edges)):
            if edge_life_index[j]:
                live_edge_set.add(edges[j])

        return live_edge_set

    def is_live_edge(self, live_edge_set, edge):
        return edge in live_edge_set or edge[::-1] in live_edge_set

    def compute_connected_component_via_bfs(self, live_edge_set, v):
        graph = self.params['network']

        visited_nodes = set()
        bfs_queue = {v}
        connected_component = set()

        while bfs_queue:
            node_to_visit = bfs_queue.pop()
            visited_nodes.add(node_to_visit)
            connected_component.add(node_to_visit)

            for u in graph.neighbors(node_to_visit):
                if u not in visited_nodes and self.is_live_edge(live_edge_set, (node_to_visit, u)):
                    bfs_queue.add(u)

        return connected_component

    def generate_connected_components(self, i):
        graph = self.params['network']
        live_edge_set = self.generate_live_edge_set()

        nodes = set(graph.nodes())
        connected_components = []
        while len(nodes) != 0:
            candidate_node = nodes.pop()
            connected_component = self.compute_connected_component_via_bfs(live_edge_set, candidate_node)
            nodes.difference_update(connected_component)
            connected_components.append(connected_component)

        dump_folder_address = root_data_address + 'sparsified_graphs/' + network_id + '/'
        os.makedirs(os.path.dirname(dump_folder_address), exist_ok = True)

        pickle.dump(connected_components, open(dump_folder_address
                                            + 'sparsified_graph_' + str(i) 
                                            + '.pkl', 'wb'))

    def generate_sparsified_graphs_by_batch(self, batch_size, batch_id, 
                                            MULTIPROCESS_BATCH = True, 
                                            num_batch_cpu = 28):
        sparsified_graphs_id_list = [(batch_size * batch_id + i) for i in range(batch_size)]

        if MULTIPROCESS_BATCH:
            with Multipool(processes=num_batch_cpu) as pool:
                pool.map(self.generate_connected_components, sparsified_graphs_id_list)
        else:
            for sparsified_graphs_id in sparsified_graphs_id_list:
                self.generate_connected_components(sparsified_graphs_id)

    def spread(self, node, sparsified_graph_id):
        connected_components = pickle.load(open(root_data_address 
                                                + 'sparsified_graphs/'
                                                + 'sparsified_graph_' + str(i) 
                                                + '.pkl', 'rb'))

        for component in connected_components:
            if node in component:
                return component


class IndependentCascadeSpreadQuerySeeding(IndependentCascade):
    def __init__(self, params):
        super(IndependentCascadeSpreadQuerySeeding, self).__init__(params)

    def query(self):
        sampled_nodes = self.params['sampled_nodes']
        all_spreads = []
        sparsified_graph_id = self.params['sparsified_graph_id']

        for i in range(self.params['k']):
            spreads = []
            infected_nodes = sampled_nodes[i][:int(self.params['rho'])]

            for node in infected_nodes:
                spreads.append(self.spread(node, sparsified_graph_id))
                sparsified_graph_id += 1

            all_spreads.append(spreads)

        return all_spreads

    def seed(self):
        all_spreads = self.query()
        seeds = []

        for i in range(self.params['k']):
            spreads = all_spreads[i]
            for j in  range(len(spreads)):
                if spreads[j].intersection(set(seeds)):
                    spreads[j] = set()
        
            candidate_score = {}
            for spread in spreads:
                for node in spread:
                    if node not in set(seeds):
                        candidate_score[node] = candidate_score.get(node, 0) + 1

            seeds.append(max(candidate_score, key = lambda node : candidate_score[node]))

        del(all_spreads)
        return seeds


class IndependentCascadeEdgeQuerySeeding(IndependentCascade):
    def __init__(self, params):
        super(IndependentCascadeEdgeQuerySeeding, self).__init__(params)

    def query(self):
        sampled_nodes = np.random.choice(list(self.params['network'].nodes()),
                                         size = int(self.params['rho']),
                                         replace = False)
        all_spreads = []
        spread_scores = []
        sparsified_graph_id = self.params['sparsified_graph_id']

        for i in range(self.params['T']):
            nodes_already_counted = set()

            for node in sampled_nodes:
                if node not in nodes_already_counted:
                    connected_component = self.spread(node, sparsified_graph_id)
                    all_spreads.append(connected_component)
                    spread_scores.append(len(connected_component.intersection(set(sampled_nodes))))
                    nodes_already_counted.update(connected_component)

            sparsified_graph_id += 1

        return all_spreads, spread_scores

    def seed(self):
        pass
