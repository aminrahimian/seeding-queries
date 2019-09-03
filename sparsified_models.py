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

    def evaluate_seeds(self, first_sparsified_graph_id, 
                    sample_size = 1000, num_sample_cpus = 28, MULTIPROCESS_SAMPLE = True):
        seeds = self.seed(first_sparsified_graph_id)
        spreads = []
        first_eval_sparsified_graph_id = self.params['eval_sparsified_graph_id']

        if MULTIPROCESS_SAMPLE:
            partial_get_spread = partial(self.get_spread_for_seed_set, seeds = seeds)
            with Multipool(processes = num_sample_cpus) as pool:
                spreads = pool.map(partial_get_spread,
                                    list(range(first_eval_sparsified_graph_id, first_eval_sparsified_graph_id + sample_size)))
        else:
            for i in range(first_eval_sparsified_graph_id, first_eval_sparsified_graph_id + sample_size):
                spreads.append(len(self.get_spread_for_seed_set(i, seeds)))

        return spreads

    def evaluate_model(self, seed_sample_size = 50, sample_size = 500, 
                       num_sample_cpus = 28, MULTIPROCESS_SAMPLE = True):
        sparsified_graph_id = self.params['sparsified_graph_id']
        graph_id_interval = self.params['k'] * int(self.params['rho'])
        graph_id_list = [sparsified_graph_id + i * graph_id_interval for i in range(seed_sample_size)]
        all_spreads = []

        for graph_id in graph_id_list:
            all_spreads += self.evaluate_seeds(graph_id,
                                               sample_size = sample_size, 
                                               num_sample_cpus = num_sample_cpus, 
                                               MULTIPROCESS_SAMPLE = MULTIPROCESS_SAMPLE)

        return np.mean(all_spreads), np.std(all_spreads), np.sum([spread < 10 for spread in all_spreads])


class IndependentCascade(ContagionModel):
    def __init__(self, params):
        super(IndependentCascade, self).__init__(params)
        

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
