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

    def spread(self, node, sparsified_graph_id):
        connected_components = pickle.load(open(root_data_address 
                                                + 'sparsified_graphs/'
                                                + self.params['network_id']
                                                + '/sparsified_graph_' + str(sparsified_graph_id) 
                                                + '.pkl', 'rb'))

        for component in connected_components:
            if node in component:
                return component

    def seed(self, first_sparsified_graph_id):
        pass

    def get_spread_for_seed_set(self, seeds, sparsified_graph_id):
        spread = set()

        for seed in seeds:
            spread.update(self.spread(seed, sparsified_graph_id))

        return spread


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

        dump_folder_address = root_data_address + 'sparsified_graphs/' + self.params['network_id'] + '/'
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


class IndependentCascadeSpreadQuerySeeding(IndependentCascade):
    def __init__(self, params):
        super(IndependentCascadeSpreadQuerySeeding, self).__init__(params)

    def query(self, first_sparsified_graph_id):
        sampled_nodes = self.params['sampled_nodes']
        all_spreads = []
        sparsified_graph_id = first_sparsified_graph_id

        for i in range(self.params['k']):
            spreads = []
            infected_nodes = sampled_nodes[i][:int(self.params['rho'])]

            for node in infected_nodes:
                spreads.append(self.spread(node, sparsified_graph_id))
                sparsified_graph_id += 1

            all_spreads.append(spreads)

        return all_spreads

    def seed(self, first_sparsified_graph_id):
        all_spreads = self.query(first_sparsified_graph_id)
        seeds = []

        for i in range(self.params['k']):
            spreads = all_spreads[i]
            for j in range(len(spreads)):
                if len(spreads[j].intersection(set(seeds))) != 0:
                    spreads[j] = set()
        
            candidate_score = {}
            for spread in spreads:
                for node in spread:
                    candidate_score[node] = candidate_score.get(node, 0) + 1

            seeds.append(max(candidate_score, key = lambda node : candidate_score[node]))

        del(all_spreads)
        return seeds


class IndependentCascadeEdgeQuerySeeding(IndependentCascade):
    def __init__(self, params):
        super(IndependentCascadeEdgeQuerySeeding, self).__init__(params)

    def query(self, first_sparsified_graph_id):
        sampled_nodes = np.random.choice(list(self.params['network'].nodes()),
                                         size = int(self.params['rho']),
                                         replace = False)
        all_spreads = []
        spread_scores = []
        sparsified_graph_id = first_sparsified_graph_id

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

    def seed(self, first_sparsified_graph_id):
        candidate_sample_size = int((self.params['size'] / self.params['k']) * np.log(1 / self.params['eps_prime']))
        all_spreads, spread_scores = self.query(first_sparsified_graph_id)
        seeds = []

        for i in range(self.params['k']):
            candidate_nodes = set(np.random.choice(list(self.params['network'].nodes()),
                                                   size = candidate_sample_size,
                                                   replace = False))

            candidate_scores = {}
            for j in range(len(all_spreads)):
                for node in all_spreads[j]:
                    if node in candidate_nodes:
                        candidate_scores[node] = candidate_scores.get(node, 0) + spread_scores[j]

            new_seed = max(candidate_scores, key = lambda node : candidate_scores[node])
            seeds.append(new_seed)
            for j in range(len(all_spreads)):
                if new_seed in all_spreads[j]:
                    spread_scores[j] = 0

        del(all_spreads)
        del(spread_scores)
        return seeds
