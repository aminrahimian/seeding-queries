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

        return None

    def seed(self, first_sparsified_graph_id):
        pass

    def get_spread_for_seed_set(self, sparsified_graph_id, seeds):
        spread = set()

        for seed in seeds:
            spread.update(self.spread(seed, sparsified_graph_id))

        return spread

    def evaluate_seeds(self, 
                       first_sparsified_graph_id, 
                       sample_size = 1000, 
                       num_sample_cpus = 7, MULTIPROCESS_SAMPLE = True):
        seeds, \
            edge_cost_with_leaf, edge_cost_without_leaf, \
            node_cost_with_leaf, node_cost_without_leaf = self.seed(first_sparsified_graph_id)
        spreads = []
        first_eval_sparsified_graph_id = self.params['eval_sparsified_graph_id']

        if not MULTIPROCESS_SAMPLE:
            for i in range(first_eval_sparsified_graph_id, first_eval_sparsified_graph_id + sample_size):
                spreads.append(len(self.get_spread_for_seed_set(i, seeds)))
        else:
            partial_get_spread = partial(self.get_spread_for_seed_set, seeds = seeds)
            with Multipool(processes = num_sample_cpus) as pool:
                spreads = pool.map(partial_get_spread,
                                   list(range(first_eval_sparsified_graph_id, first_eval_sparsified_graph_id + sample_size)))
                spreads = [len(spread) for spread in spreads]

        return spreads, (edge_cost_with_leaf, edge_cost_without_leaf, node_cost_with_leaf, node_cost_without_leaf)

    def evaluate_model(self, seed_sample_size = 50, sample_size = 500, 
                       num_seed_sample_cpus = 4, MULTIPROCESS_SEED_SAMPLE = True,
                       num_sample_cpus = 7, MULTIPROCESS_SAMPLE = True):
        sparsified_graph_id = self.params['sparsified_graph_id']
        graph_id_interval = self.params['graph_id_interval']
        graph_id_list = [sparsified_graph_id + i * graph_id_interval for i in range(seed_sample_size)]
        all_spreads = []
        all_costs = []

        if not MULTIPROCESS_SEED_SAMPLE:
            spread_items = []
            for graph_id in graph_id_list:
                spread_items.append(self.evaluate_seeds(graph_id, 
                                                        sample_size = sample_size,
                                                        num_sample_cpus = num_sample_cpus,
                                                        MULTIPROCESS_SAMPLE = MULTIPROCESS_SAMPLE))
            for spread_item in spread_items:
                all_spreads += spread_item[0]
                all_costs.append(spread_item[1])
        else:
            partial_eval_seeds = partial(self.evaluate_seeds, 
                                         sample_size = sample_size,
                                         num_sample_cpus = num_sample_cpus,
                                         MULTIPROCESS_SAMPLE = MULTIPROCESS_SAMPLE)
            with Multipool(processes = num_seed_sample_cpus) as pool:
                spread_items = pool.map(partial_eval_seeds, graph_id_list)
            for spread_item in spread_items:
                all_spreads += spread_item[0]
                all_costs.append(spread_item[1])

        return all_spreads, all_costs


class IndependentCascade(ContagionModel):
    def __init__(self, params):
        super(IndependentCascade, self).__init__(params)
        

class IndependentCascadeSpreadQuerySeeding(IndependentCascade):
    def __init__(self, params):
        super(IndependentCascadeSpreadQuerySeeding, self).__init__(params)

    def query(self, first_sparsified_graph_id):
        order_id = int((first_sparsified_graph_id - self.params['sparsified_graph_id']) / self.params['graph_id_interval'])
        sampled_nodes = self.params['sampled_nodes'][order_id]
        all_spreads = []

        unique_node_set_with_leaf, unique_node_set_without_leaf = set(), set()
        unique_edge_set_with_leaf, unique_edge_set_without_leaf = set(), set()

        for i in range(self.params['k']):
            spreads = []
            infected_nodes = sampled_nodes[i][:int(self.params['rho'])]
            sparsified_graph_id = first_sparsified_graph_id + i * int(self.params['graph_id_interval'] / self.params['k'])

            for node in infected_nodes:
                spread = self.spread(node, sparsified_graph_id)
                spreads.append(spread)

                subgraph = G.subgraph(spread)

                unique_node_set_with_leaf.update(spread)
                unique_node_set_without_leaf.update(set(filter(lambda node : subgraph.degree(node) > 1, 
                                                               set(subgraph.nodes()))))

                unique_edge_set_with_leaf.update(set(subgraph.edges()))
                unique_edge_set_without_leaf.update(set(filter(lambda edge : subgraph.degree(edge[0]) > 1 and subgraph.degree(edge[1]) > 1, 
                                                               set(subgraph.edges()))))

                sparsified_graph_id += 1

            all_spreads.append(spreads)

        return (all_spreads, 
                len(unique_edge_set_with_leaf),
                len(unique_edge_set_without_leaf),
                len(unique_node_set_with_leaf),
                len(unique_node_set_without_leaf))

    def seed(self, first_sparsified_graph_id):
        all_spreads, \
            edge_cost_with_leaf, edge_cost_without_leaf, \
            node_cost_with_leaf, node_cost_without_leaf = self.query(first_sparsified_graph_id)
        seeds = []
        order_id = int((first_sparsified_graph_id - self.params['sparsified_graph_id']) / self.params['graph_id_interval'])
        candidate_nodes = self.params['candidate_nodes'][order_id]

        for i in range(self.params['k']):
            spreads = all_spreads[i]      
            candidate_score = {}
            for spread in spreads:
                for node in spread:
                    candidate_score[node] = candidate_score.get(node, 0) + 1

            if len(candidate_score) == 0:
                for sampled_node in candidate_nodes:
                    if sampled_node not in seeds:
                        new_seed = sampled_node
                        break
            else:
                candidate_by_score = {}
                for candidate in candidate_score:
                    if candidate_score[candidate] not in candidate_by_score:
                        candidate_by_score[candidate_score[candidate]] = set()
                    candidate_by_score[candidate_score[candidate]].add(candidate)

                max_score = max(candidate_by_score)
                top_candidates = candidate_by_score[max_score].difference(set(seeds))
                new_seed = min(top_candidates, key = lambda x : candidate_nodes.index(x))

            seeds.append(new_seed)
            for j in range(len(spreads)):
                if new_seed in spreads[j]:
                    spreads[j] = set()

        del(all_spreads)
        return (seeds, 
                edge_cost_with_leaf, 
                edge_cost_without_leaf,
                node_cost_with_leaf,
                node_cost_without_leaf)


class IndependentCascadeEdgeQuerySeeding(IndependentCascade):
    def __init__(self, params):
        super(IndependentCascadeEdgeQuerySeeding, self).__init__(params)

    def query(self, first_sparsified_graph_id):
        order_id = int((first_sparsified_graph_id - self.params['sparsified_graph_id']) / self.params['graph_id_interval'])
        sampled_nodes = self.params['sampled_nodes'][order_id][:int(self.params['rho'])]
        
        all_spreads = []
        spread_scores = []
        sparsified_graph_id = first_sparsified_graph_id

        unique_node_set_with_leaf, unique_node_set_without_leaf = set(), set()
        unique_edge_set_with_leaf, unique_edge_set_without_leaf = set(), set()

        for i in range(self.params['T']):
            nodes_already_counted = set()

            for node in sampled_nodes:
                if node not in nodes_already_counted:
                    connected_component = self.spread(node, sparsified_graph_id)

                    subgraph = G.subgraph(connected_component)

                    unique_node_set_with_leaf.update(connected_component)
                    unique_node_set_without_leaf.update(set(filter(lambda node : subgraph.degree(node) > 1, 
                                                                set(subgraph.nodes()))))

                    unique_edge_set_with_leaf.update(set(subgraph.edges()))
                    unique_edge_set_without_leaf.update(set(filter(lambda edge : subgraph.degree(edge[0]) > 1 and subgraph.degree(edge[1]) > 1, 
                                                                set(subgraph.edges()))))

                    all_spreads.append(connected_component)
                    spread_scores.append(len(connected_component.intersection(set(sampled_nodes))))
                    nodes_already_counted.update(connected_component)

            sparsified_graph_id += 1

        return (all_spreads, 
                spread_scores,
                len(unique_edge_set_with_leaf),
                len(unique_edge_set_without_leaf),
                len(unique_node_set_with_leaf),
                len(unique_node_set_without_leaf))

    def seed(self, first_sparsified_graph_id):
        all_spreads, spread_scores, \
            edge_cost_with_leaf, edge_cost_without_leaf, \
            node_cost_with_leaf, node_cost_without_leaf = self.query(first_sparsified_graph_id)
        seeds = []
        order_id = int((first_sparsified_graph_id - self.params['sparsified_graph_id']) / self.params['graph_id_interval'])
        candidate_nodes = self.params['candidate_nodes'][order_id]

        for i in range(self.params['k']):
            candidate_scores = {}
            for j in range(len(all_spreads)):
                for node in all_spreads[j]:
                    candidate_scores[node] = candidate_scores.get(node, 0) + spread_scores[j]
            
            if len(candidate_scores) == 0:
               for node in candidate_nodes:
                   if node not in seeds:
                       new_seed = node
                       break
            else:
                candidate_by_score = {}
                for candidate in candidate_scores:
                    if candidate_scores[candidate] not in candidate_by_score:
                        candidate_by_score[candidate_scores[candidate]] = set()
                    candidate_by_score[candidate_scores[candidate]].add(candidate)

                max_score = max(candidate_by_score)
                top_candidates = candidate_by_score[max_score].difference(set(seeds))
                new_seed = min(top_candidates, key = lambda x : candidate_nodes.index(x))
            
            seeds.append(new_seed)
            for j in range(len(all_spreads)):
                if new_seed in all_spreads[j]:
                    spread_scores[j] = 0

        del(all_spreads)
        del(spread_scores)
        return (seeds,
                edge_cost_with_leaf, 
                edge_cost_without_leaf,
                node_cost_with_leaf,
                node_cost_without_leaf)
