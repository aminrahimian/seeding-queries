from settings import *
import gc
import copy
import itertools
import heapq

from functools import partial

from Multipool import Multipool

TRACK_TIME_SINCE_VARIABLES = True


def measure_property(network_intervention_dataset, property='avg_clustering', sample_size=None):
    if sample_size is not None:
        assert sample_size <= len(network_intervention_dataset), \
            "not enough samples to do measurements on network_intervention_dataset, sample_size: " + str(sample_size) \
            + "len(network_intervention_dataset): " + str(len(network_intervention_dataset))
        network_intervention_dataset = network_intervention_dataset[0:sample_size - 1]

    property_samples = []

    for network_intervention in network_intervention_dataset:
        if property is 'avg_clustering':
            property_sample = NX.average_clustering(network_intervention)
        elif property is 'average_shortest_path_length':
            property_sample = NX.average_shortest_path_length(network_intervention)
        elif property is 'diameter':
            property_sample = NX.diameter(network_intervention)
        elif property is 'size_2_core':
            sample_2_core = NX.k_core(network_intervention, k=2)
            property_sample = sample_2_core.number_of_nodes()
        elif property is 'avg_degree':
            degree_sequence = [d for n, d in network_intervention.degree()]
            sum_of_edges = sum(degree_sequence)
            property_sample = sum_of_edges / network_intervention.number_of_nodes()
        elif property is 'diam_2_core':
            sample_2_core = NX.k_core(network_intervention, k=2)
            if sample_2_core.number_of_nodes() == 0:
                property_sample = float('Inf')
            else:
                property_sample = NX.diameter(sample_2_core)
        elif property is 'max_degree':
            degree_sequence = sorted([d for n, d in network_intervention.degree()], reverse=True)
            property_sample = max(degree_sequence)
        elif property is 'min_degree':
            degree_sequence = sorted([d for n, d in network_intervention.degree()], reverse=True)
            property_sample = min(degree_sequence)
        elif property is 'max_degree_2_core':
            sample_2_core = NX.k_core(network_intervention, k=2)
            if sample_2_core.number_of_nodes() == 0:
                property_sample = 0
            else:
                degree_sequence = sorted([d for n, d in sample_2_core.degree()], reverse=True)
                property_sample = max(degree_sequence)
        elif property is 'min_degree_2_core':
            sample_2_core = NX.k_core(network_intervention, k=2)
            if sample_2_core.number_of_nodes() == 0:
                property_sample = 0
            else:
                degree_sequence = sorted([d for n, d in sample_2_core.degree()], reverse=True)
                property_sample = min(degree_sequence)
        elif property is 'avg_degree_2_core':
            sample_2_core = NX.k_core(network_intervention, k=2)
            if sample_2_core.number_of_nodes() == 0:
                property_sample = 0
            else:
                degree_sequence = [d for n, d in sample_2_core.degree()]
                sum_of_edges = sum(degree_sequence)
                property_sample = sum_of_edges / sample_2_core.number_of_nodes()
        elif property is 'number_edges':
            property_sample = network_intervention.number_of_edges()
        elif property is 'number_edges_2_core':
            sample_2_core = NX.k_core(network_intervention, k=2)
            if sample_2_core.number_of_nodes() == 0:
                property_sample = 0
            else:
                property_sample = sample_2_core.number_of_edges()
        elif property is 'avg_clustering_2_core':
            sample_2_core = NX.k_core(network_intervention, k=2)
            if sample_2_core.number_of_nodes() == 0:
                property_sample = float('NaN')
            else:
                property_sample = NX.average_clustering(sample_2_core)
        elif property is 'transitivity':
            property_sample = NX.transitivity(network_intervention)
        elif property is 'transitivity_2_core':
            sample_2_core = NX.k_core(network_intervention, k=2)
            if sample_2_core.number_of_nodes() == 0:
                property_sample = float('NaN')
            else:
                property_sample = NX.transitivity(sample_2_core)
        elif property is 'num_leaves':

            degree_sequence = sorted([d for n, d in network_intervention.degree()], reverse=True)
            property_sample = degree_sequence.count(int(1))
        else:
            assert False, property + ' property not supported.'

        property_samples += [property_sample]

    return property_samples


def random_factor_pair(value):
    """
    Returns a random pair that factor value.
    It is used to set the number of columns and rows in  2D grid with a given size such that
    size = num_columns*num_rows
    """
    factors = []
    for i in range(1, int(value ** 0.5) + 1):
        if value % i == 0:
            factors.append((int(i), value // i))
    return RD.choice(factors)


def newman_watts_add_fixed_number_graph(n, k=2, p=2, seed=None):
    """ Returns a Newman-Watts-Strogatz small-world graph. With a fixed - p - (not random)
    number of edges added to each node. Modified newman_watts_strogatzr_graph() in NetworkX. """
    if seed is not None:
        RD.seed(seed)
    if k >= n:
        raise NX.NetworkXError("k>=n, choose smaller k or larger n")
    G = NX.connected_watts_strogatz_graph(n, k, 0)

    all_nodes = G.nodes()
    for u in all_nodes:
        count_added_edges = 0  # track number of edges added to node u
        while count_added_edges < p:

            w = np.random.choice(all_nodes)

            # no self-loops and reject if edge u-w exists
            # is that the correct NWS model?
            while w == u or G.has_edge(u, w):
                w = np.random.choice(all_nodes)
                # print('re-drawn w', w)
                if G.degree(u) >= n - 1:
                    break  # skip this rewiring
            G.add_edge(u, w)
            count_added_edges += 1
    return G


def cycle_union_erdos_renyi(n, k=4, c=2, seed=None,
                            color_the_edges=False,
                            cycle_edge_color='k',
                            random_edge_color='b',
                            weight_the_edges=False,
                            cycle_edge_weights=4,
                            random_edge_weights=4):
    """Returns a cycle C_k union G(n,c/n) graph by composing
    NX.connected_watts_strogatz_graph(n, k, 0) and
    NX.erdos_renyi_graph(n, c/n, seed=None, directed=False)"""
    if seed is not None:
        RD.seed(seed)

    if k >= n:
        raise NX.NetworkXError("k>=n, choose smaller k or larger n")
    C_k = NX.connected_watts_strogatz_graph(n, k, 0)
    if color_the_edges:
        # cycle_edge_colors = dict.fromkeys(C_k.edges(), cycle_edge_color)
        NX.set_edge_attributes(C_k, cycle_edge_color, 'color')
    if weight_the_edges:
        NX.set_edge_attributes(C_k, cycle_edge_weights, 'weight')

    G_npn = NX.erdos_renyi_graph(n, c / n, seed=None, directed=False)
    if color_the_edges:
        # random_edge_colors = dict.fromkeys(G_npn.edges(), random_edge_color)
        NX.set_edge_attributes(G_npn, random_edge_color, 'color')
    if weight_the_edges:
        NX.set_edge_attributes(G_npn, random_edge_weights, 'weight')

    assert G_npn.nodes() == C_k.nodes(), "node sets are not the same"
    composed = NX.compose(G_npn, C_k)

    # print(composed.edges.data())

    return composed


def c_1_c_2_interpolation(n, eta, add_long_ties_exp, remove_cycle_edges_exp, seed=None):
    """Return graph that interpolates C_1 and C_2.
    Those edges having add_long_ties_exp < eta are added.
    Those edges having remove_cycle_edges_exp < eta are removed.
    len(add_long_ties_exp) = n*(n-1)//2
    len(remove_cycle_edges_exp) = n
    """
    if seed is not None:
        RD.seed(seed)

    assert len(add_long_ties_exp) == n * (n - 1) // 2, "add_long_ties_exp has the wrong size"
    assert len(remove_cycle_edges_exp) == n, "remove_cycle_edges_exp has the wrong size"

    C_2 = NX.connected_watts_strogatz_graph(n, 4, 0)

    C_2_minus_C_1_edge_index = 0
    removal_list = []

    for edge in C_2.edges():
        # print(edge)
        if abs(edge[0] - edge[1]) == 2 or abs(edge[0] - edge[1]) == n - 2:  # it is a C_2\C_1 edge
            if remove_cycle_edges_exp[C_2_minus_C_1_edge_index] < eta:
                removal_list += [edge]
            C_2_minus_C_1_edge_index += 1  # index the next C_2\C_1 edge
    C_2.remove_edges_from(removal_list)

    addition_list = []

    K_n = NX.complete_graph(n)

    random_add_edge_index = 0

    for edge in K_n.edges():

        if add_long_ties_exp[random_add_edge_index] < eta:
            addition_list += [edge]
        random_add_edge_index += 1  # index the next edge to be considered for addition
    C_2.add_edges_from(addition_list)

    return C_2


def add_edges(G, number_of_edges_to_be_added=10, mode='random', seed=None):
    """Add number_of_edges_to_be_added edges to the NetworkX object G.
    Two modes: 'random' or 'triadic_closures'
    """
    if seed is not None:
        RD.seed(seed)

    number_of_edges_to_be_added = int(np.floor(number_of_edges_to_be_added))

    assert type(G) is NX.classes.graph.Graph, "input should be a NetworkX graph object"

    fat_network = copy.deepcopy(G)
    unformed_edges = list(NX.non_edges(fat_network))

    if len(unformed_edges) < number_of_edges_to_be_added:
        print("There are not that many edges left ot be added")
        fat_network.add_edges_from(unformed_edges)  # add all the edges that are left
        return fat_network

    if mode is 'random':
        addition_list = RD.sample(unformed_edges, number_of_edges_to_be_added)
        fat_network.add_edges_from(addition_list)
        return fat_network

    if mode is 'triadic_closures':
        weights = []
        for non_edge in unformed_edges:
            weights += [1.0 * len(list(NX.common_neighbors(G, non_edge[0], non_edge[1])))]

        total_sum = sum(weights)

        normalized_weights = [weight / total_sum for weight in weights]

        addition_list = np.random.choice(range(len(unformed_edges)),
                                         number_of_edges_to_be_added,
                                         replace=False,
                                         p=normalized_weights)

        addition_list = addition_list.astype(int)

        addition_list = [unformed_edges[ii] for ii in list(addition_list)]

        fat_network.add_edges_from(addition_list)

        return fat_network


def maslov_sneppen_rewiring(G, num_steps=SENTINEL, return_connected=True, seed=None):
    """
    Rewire the network graph according to the Maslov and
    Sneppen method for degree-preserving random rewiring of a complex network,
    as described on
    `Maslov's webpage <http://www.cmth.bnl.gov/~maslov/matlab.htm>`_.
    Return the resulting graph.
    If a positive integer ``num_steps`` is given, then perform ``num_steps``
    number of steps of the method.
    Otherwise perform the default number of steps of the method, namely
    ``4*graph.num_edges()`` steps.
    The code is adopted from: https://github.com/araichev/graph_dynamics/blob/master/graph_dynamics.py
    """
    if seed is not None:
        RD.seed(seed)

    assert type(G) is NX.classes.graph.Graph, "input should be a NetworkX graph object"

    if num_steps is SENTINEL:
        num_steps = 10 * G.number_of_edges()
        # completely rewire everything

    rewired_network = copy.deepcopy(G)
    for i in range(num_steps):
        chosen_edges = RD.sample(rewired_network.edges(), 2)
        e1 = chosen_edges[0]
        e2 = chosen_edges[1]
        new_e1 = (e1[0], e2[1])
        new_e2 = (e2[0], e1[1])
        if new_e1[0] == new_e1[1] or new_e2[0] == new_e2[1] or \
                rewired_network.has_edge(*new_e1) or rewired_network.has_edge(*new_e2):
            # Not allowed to rewire e1 and e2. Skip.
            continue

        rewired_network.remove_edge(*e1)
        rewired_network.remove_edge(*e2)
        rewired_network.add_edge(*new_e1)
        rewired_network.add_edge(*new_e2)

    if return_connected:
        while not NX.is_connected(rewired_network):
            rewired_network = copy.deepcopy(G)
            for i in range(num_steps):
                chosen_edges = RD.sample(rewired_network.edges(), 2)
                e1 = chosen_edges[0]
                e2 = chosen_edges[1]
                new_e1 = (e1[0], e2[1])
                new_e2 = (e2[0], e1[1])
                if new_e1[0] == new_e1[1] or new_e2[0] == new_e2[1] or \
                        rewired_network.has_edge(*new_e1) or rewired_network.has_edge(*new_e2):
                    # Not allowed to rewire e1 and e2. Skip.
                    continue

                rewired_network.remove_edge(*e1)
                rewired_network.remove_edge(*e2)
                rewired_network.add_edge(*new_e1)
                rewired_network.add_edge(*new_e2)

    return rewired_network


def random_random_rewiring(G, num_edges=SENTINEL, return_connected=True, seed=None):
    """
    Rewire the network graph.
    Choose num_edges randomly from the existing edges and remove them.
    Choose num_edges randomly from the non-existing edges and add them.
    """
    if seed is not None:
        RD.seed(seed)

    assert type(G) is NX.classes.graph.Graph, "input should be a NetworkX graph object"

    if num_edges is SENTINEL:
        num_edges = G.number_of_edges()
        print('Warning: number of edges to rewire not supplied, all edges will be rewired.')
        # completely rewire everything

    rewired_network = copy.deepcopy(G)

    unformed_edges = list(NX.non_edges(rewired_network))

    formed_edges = list(NX.edges(rewired_network))

    addition_list = np.random.choice(range(len(unformed_edges)),
                                     num_edges,
                                     replace=False)

    addition_list = addition_list.astype(int)

    addition_list = [unformed_edges[ii] for ii in list(addition_list)]

    rewired_network.add_edges_from(addition_list)

    removal_list = np.random.choice(range(len(formed_edges)),
                                    num_edges,
                                    replace=False)

    removal_list = removal_list.astype(int)

    removal_list = [formed_edges[ii] for ii in list(removal_list)]

    rewired_network.remove_edges_from(removal_list)

    if return_connected:
        while not NX.is_connected(rewired_network):
            rewired_network = copy.deepcopy(G)

            unformed_edges = list(NX.non_edges(rewired_network))

            formed_edges = list(NX.edges(rewired_network))

            addition_list = np.random.choice(range(len(unformed_edges)),
                                             num_edges,
                                             replace=False)

            addition_list = addition_list.astype(int)

            addition_list = [unformed_edges[ii] for ii in list(addition_list)]

            rewired_network.add_edges_from(addition_list)

            removal_list = np.random.choice(range(len(formed_edges)),
                                            num_edges,
                                            replace=False)

            removal_list = removal_list.astype(int)

            removal_list = [formed_edges[ii] for ii in list(removal_list)]

            rewired_network.remove_edges_from(removal_list)

    return rewired_network


class ContagionModel(object):
    """
    implement the initializations and parameter set methods
    """

    def __init__(self,params):
        self.network_initialized = False
        self.states_initialized = False

        self.fixed_params = copy.deepcopy(params)
        self.params = params
        self.missing_params_not_set = True
        self.number_of_active_infected_neighbors_is_updated = False
        self.time_since_infection_is_updated = False
        self.time_since_activation_is_updated = False
        self.list_of_susceptible_agents_is_updated = False
        self.list_of_active_infected_agents_is_updated = False
        self.list_of_inactive_infected_agents_is_updated = False
        self.number_of_active_infected_neighbors_is_updated = False
        self.list_of_most_recent_activations_is_updated = False

    def init_network(self):
        """
        initializes the network interconnections based on the params
        """
        if 'network' in self.fixed_params:
            self.params['network'] = self.fixed_params['network']
        elif 'network' not in self.fixed_params:
            if self.params['network_model'] == 'erdos_renyi':
                if 'linkProbability' not in self.fixed_params:  # erdos-renyi link probability
                    self.params['linkProbability'] = 2 * np.log(self.params['size']) / self.params[
                        'size']  # np.random.beta(1, 1, None)*20*np.log(self.params['size'])/self.params['size']
                self.params['network'] = NX.erdos_renyi_graph(self.params['size'], self.params['linkProbability'])
                if not NX.is_connected(self.params['network']):
                    self.params['network'] = NX.erdos_renyi_graph(self.params['size'], self.params['linkProbability'])

            elif self.params['network_model'] == 'watts_strogatz':
                if 'nearest_neighbors' not in self.fixed_params:
                    self.params['nearest_neighbors'] = 3
                if 'rewiring_probability' not in self.fixed_params:
                    self.params['rewiring_probability'] = 0.000000005
                self.params['network'] = NX.connected_watts_strogatz_graph(self.params['size'],
                                                                           self.params['nearest_neighbors'],
                                                                           self.params['rewiring_probability'])
            elif self.params['network_model'] == 'grid':
                if 'number_grid_rows' not in self.fixed_params:
                    if 'number_grid_columns' not in self.fixed_params:
                        (self.params['number_grid_columns'], self.params['number_grid_rows']) = \
                            random_factor_pair(self.params['size'])
                    else:
                        self.params['number_grid_rows'] = self.params['size'] // self.params['number_grid_columns']
                        self.params['number_grid_columns'] = self.params['size'] // self.params['number_grid_rows']
                elif 'number_grid_columns' in self.fixed_params:
                    assert self.params['number_grid_columns'] * self.params['number_grid_rows'] == self.params['size'], \
                        'incompatible size and grid dimensions'
                else:
                    self.params['number_grid_columns'] = self.params['size'] // self.params['number_grid_rows']
                    self.params['number_grid_rows'] = self.params['size'] // self.params['number_grid_columns']
                self.params['network'] = NX.grid_2d_graph(self.params['number_grid_rows'],
                                                          self.params['number_grid_columns'])
            elif self.params['network_model'] == 'random_regular':
                if 'degree' not in self.fixed_params:
                    self.params['degree'] = np.random.randint(1, 6)
                self.params['network'] = NX.random_regular_graph(self.params['degree'], self.params['size'], seed=None)
            elif self.params['network_model'] == 'newman_watts_fixed_number':
                if 'fixed_number_edges_added' not in self.fixed_params:
                    self.params['fixed_number_edges_added'] = 2
                if 'nearest_neighbors' not in self.fixed_params:
                    self.params['nearest_neighbors'] = 2
                self.params['network'] = newman_watts_add_fixed_number_graph(self.params['size'],
                                                                             self.params['nearest_neighbors'],
                                                                             self.params['fixed_number_edges_added'])
            elif self.params['network_model'] == 'cycle_union_Erdos_Renyi':
                if 'c' not in self.fixed_params:
                    self.params['c'] = 2
                if 'nearest_neighbors' not in self.fixed_params:
                    self.params['nearest_neighbors'] = 2
                self.params['network'] = cycle_union_erdos_renyi(self.params['size'], self.params['nearest_neighbors'],
                                                                 self.params['c'])

            elif self.params['network_model'] == 'c_1_c_2_interpolation':
                if 'c' not in self.fixed_params:
                    self.params['c'] = 2
                if 'nearest_neighbors' not in self.fixed_params:
                    self.params['nearest_neighbors'] = 2
                if 'add_long_ties_exp' not in self.fixed_params:
                    self.params['add_long_ties_exp'] = np.random.exponential(scale=self.params['size'] ** 2,
                                                                             size=int(1.0 * self.params['size']
                                                                                      * (self.params['size'] - 1)) // 2)

                    self.params['remove_cycle_edges_exp'] = np.random.exponential(scale=2 * self.params['size'],
                                                                                  size=self.params['size'])

                self.params['network'] = c_1_c_2_interpolation(self.params['size'], self.params['eta'],
                                                               self.params['add_long_ties_exp'],
                                                               self.params['remove_cycle_edges_exp'])
            else:
                assert False, 'undefined network type'

        # when considering real network and interventions on them we may need to record the original network.
        # This is currently only used in SimpleOnlyAlongOriginalEdges(ContagionModel)

        if 'original_network' in self.fixed_params:
            self.params['original_network'] = self.fixed_params['original_network']
        else:
            self.params['original_network'] = None

        # additional modifications / structural interventions to the network topology which include rewiring
        # and edge additions

        if 'rewire' not in self.fixed_params:
            self.params['rewire'] = False
            print('warning: the network will not be rewired!')

        if self.params['rewire']:

            if 'rewiring_mode' not in self.fixed_params:
                self.params['rewiring_mode'] = 'maslov_sneppen'
                print('warning: the rewiring mode is set to maslov_sneppen')
            if self.params['rewiring_mode'] == 'maslov_sneppen':
                if 'num_steps_for_maslov_sneppen_rewiring' not in self.fixed_params:
                    self.params['num_steps_for_maslov_sneppen_rewiring'] = \
                        0.1 * self.params['network'].number_of_edges()  # rewire 10% of edges
                    print('Warning: num_steps_for_maslov_sneppen_rewiring is set to default 10%')
                rewired_network = \
                    self.maslov_sneppen_rewiring(
                        num_steps=int(np.floor(self.params['num_steps_for_maslov_sneppen_rewiring'])))
            elif self.params['rewiring_mode'] == 'random_random':
                if 'num_edges_for_random_random_rewiring' not in self.fixed_params:
                    self.params['num_edges_for_random_random_rewiring'] = \
                        0.1 * self.params['network'].number_of_edges()  # rewire 10% of edges
                    print('warning: num_edges_for_random_random_rewiring is set to default 10%')

                rewired_network = \
                    self.random_random_rewiring(
                        num_edges=int(np.floor(self.params['num_edges_for_random_random_rewiring'])))

            self.params['network'] = rewired_network

        if 'add_edges' not in self.fixed_params:
            self.params['add_edges'] = False

        if self.params['add_edges']:
            if 'edge_addition_mode' not in self.fixed_params:
                self.params['edge_addition_mode'] = 'triadic_closures'
            if 'number_of_edges_to_be_added' not in self.fixed_params:
                self.params['number_of_edges_to_be_added'] = \
                    int(np.floor(0.15 * self.params['network'].number_of_edges()))  # add 15% more edges

            fattened_network = add_edges(self.params['network'],
                                         self.params['number_of_edges_to_be_added'],
                                         self.params['edge_addition_mode'])

            self.params['network'] = fattened_network

        self.node_list = sorted(self.params['network'].nodes())  # used for indexing nodes in cases where
        # node attributes are available in a list. A typical application is as follows: self.node_list.index(i)
        # for i in self.params['network'].nodes():
        # If your node data is not needed, it is simpler and equivalent to use the expression for n in G, or list(G).
        # instead of networkx.Graph.nodes

        self.network_initialized = True

    def init_network_states(self, initially_infected_node_indexes):

        """
                initializes the node states (infected/susceptible) and other node attributes
                such as number of infected neighbours and time since infection
                according to initially_infected_node_indexes.
                initially_infected_node_indexes will be output by the seed method
                in a seeding class that inherits NetworkModel
        """

        # this method will be overridden by a method in the seed class if inherited

        # when performing state transitions the following eight flags should be updated:
        # self.number_of_active_infected_neighbors_is_updated
        # self.time_since_infection_is_updated
        # self.time_since_activation_is_updated
        # self.list_of_susceptible_agents_is_updated
        # self.list_of_active_infected_agents_is_updated
        # self.list_of_inactive_infected_agents_is_updated
        # self.list_of_exposed_agents_is_updated
        # self.list_of_most_recent_activations_is_updated

        self.number_of_active_infected_neighbors_is_updated = False
        self.time_since_infection_is_updated = False
        self.time_since_activation_is_updated = False

        self.list_of_susceptible_agents = []
        self.list_of_susceptible_agents_is_updated = False
        self.list_of_active_infected_agents = []
        self.list_of_active_infected_agents_is_updated = False
        self.list_of_inactive_infected_agents = []
        self.list_of_inactive_infected_agents_is_updated = False
        self.list_of_exposed_agents = []
        self.list_of_exposed_agents_is_updated = False
        self.list_of_most_recent_activations = []
        self.list_of_most_recent_activations_is_updated = False
        # list of most recent activations is useful for speeding up pure (0,1)
        # complex contagion computations by shortening the loop over exposed agents.

        self.params['initial_states'] = 1.0 * np.zeros(self.params['size'])
        for node in initially_infected_node_indexes:
            self.params['initial_states'][self.node_list.index(node)] = infected * active
        # all nodes are initially active by default
        self.params['initial_states'] = list(self.params['initial_states'])

        for i in self.params['network'].nodes():
            self.params['network'].node[i]['number_of_active_infected_neighbors'] = 0
            self.params['network'].node[i]['time_since_infection'] = 0
            self.params['network'].node[i]['time_since_activation'] = 0
            self.params['network'].node[i]['threshold'] = self.params['thresholds'][self.node_list.index(i)]

        self.time_since_infection_is_updated = True
        self.time_since_activation_is_updated = True

        for i in self.params['network'].nodes():
            self.params['network'].node[i]['state'] = self.params['initial_states'][self.node_list.index(i)]
            if self.params['network'].node[i]['state'] == infected * active:
                self.list_of_active_infected_agents.append(i)
                self.list_of_most_recent_activations.append(i)
            elif self.params['network'].node[i]['state'] == infected * inactive:
                self.list_of_inactive_infected_agents.append(i)
            elif self.params['network'].node[i]['state'] == susceptible:
                self.list_of_susceptible_agents.append(i)
            else:
                print('node', i)
                print('state', self.params['network'].node[i]['state'])
                print('state initialization miss-handled')
                exit()

        self.list_of_susceptible_agents_is_updated = True
        self.list_of_active_infected_agents_is_updated = True
        self.list_of_inactive_infected_agents_is_updated = True
        self.list_of_most_recent_activations_is_updated = True

        for i in self.list_of_active_infected_agents + self.list_of_inactive_infected_agents:
            for j in self.params['network'].neighbors(i):
                self.params['network'].node[j]['number_of_active_infected_neighbors'] += 1
                if ((j not in self.list_of_exposed_agents) and
                        (self.params['network'].node[j]['state'] == susceptible)):
                    self.list_of_exposed_agents.append(j)

        self.number_of_active_infected_neighbors_is_updated = True

        self.list_of_exposed_agents_is_updated = True

        assert self.number_of_active_infected_neighbors_is_updated and \
               self.time_since_infection_is_updated and \
               self.time_since_activation_is_updated and \
               self.list_of_susceptible_agents_is_updated and \
               self.list_of_active_infected_agents_is_updated and \
               self.list_of_inactive_infected_agents_is_updated and \
               self.list_of_exposed_agents_is_updated and \
               self.list_of_most_recent_activations_is_updated, \
            'error: state lists miss handled in the initializations'

        self.updated_list_of_susceptible_agents = []
        self.updated_list_of_active_infected_agents = []
        self.updated_list_of_inactive_infected_agents = []
        self.updated_list_of_exposed_agents = []
        self.updated_list_of_most_recent_activations = []

    def random_init(self, with_seed = True):
        """
        the parameters that are provided when the class is being initialized are treated as fixed. The missing parameters
        are set randomly. In an inference framework the distributions that determine the random draws are priors over
        those parameters which are not fixed.
        """
        assert self.missing_params_not_set, 'error: missing parameters are already set.'
        # no spontaneous adoptions
        if 'zero_at_zero' not in self.fixed_params:
            self.params['zero_at_zero'] = True
        # below threshold adoption rate is divided by the self.params['multiplier']
        if 'multiplier' not in self.fixed_params:
            self.params['multiplier'] = 5
        # the high probability in complex contagion
        if 'fixed_prob_high' not in self.fixed_params:
            self.params['fixed_prob_high'] = 1.0
        # the low probability in complex contagion
        if 'fixed_prob' not in self.fixed_params:
            self.params['fixed_prob'] = 0.0
        # SI infection rate
        if 'beta' not in self.fixed_params:  # SIS infection rate parameter
            self.params['beta'] = RD.choice(
                [0.2, 0.3, 0.4, 0.5])  # 0.1 * np.random.beta(1, 1, None)#0.2 * np.random.beta(1, 1, None)
        if 'sigma' not in self.fixed_params:  # logit and probit parameter
            self.params['sigma'] = RD.choice([0.1, 0.3, 0.5, 0.7, 1])
        # complex contagion threshold
        if 'theta' not in self.fixed_params:  # complex contagion threshold parameter
            self.params['theta'] = RD.choice([1, 2, 3, 4])  # np.random.randint(1, 4)
        #  The default values gamma = 0 and alpha = 1 ensure that all infected nodes always remain active
        if 'gamma' not in self.fixed_params:  # rate of transition from active to inactive
            self.params['gamma'] = 0.0  # RD.choice([0.2,0.3,0.4,0.5])
        if 'alpha' not in self.fixed_params:  # rate of transition from inactive to active
            self.params['alpha'] = 1.0  # RD.choice([0.02,0.03,0.04,0.05])
        if 'memory' not in self.fixed_params:  # used in the independent cascade model
            self.params['memory'] = 0.0
        if 'size' not in self.fixed_params:
            if 'network' in self.fixed_params:
                self.params['size'] = NX.number_of_nodes(self.params['network'])
            else:
                self.params['size'] = 100  # np.random.randint(50, 500)
        if 'network_model' not in self.fixed_params:
            self.params['network_model'] = RD.choice(['erdos_renyi', 'watts_strogatz', 'grid', 'random_regular'])

        if 'thresholds' not in self.params:
            assert not hasattr(self, 'isLinearThresholdModel'), \
                "Thresholds should have been already set in the linear threshold model!"
            if 'thresholds' in self.fixed_params:
                self.params['thresholds'] = self.fixed_params['thresholds']
            else:
                self.params['thresholds'] = [self.params['theta']] * self.params['size']

        self.init_network()

        if with_seed:
            seed_set, node_discovery_cost, edge_discovery_cost = self.seed()
        else:
            seed_set = []

        self.init_network_states(seed_set)

        self.missing_params_not_set = False

        self.spread_stopped = False

        if with_seed:
            return node_discovery_cost, edge_discovery_cost

    def time_the_total_spread(self, cap=0.99,
                              get_time_series=False,
                              verbose=False):
        time = 0
        network_time_series = []
        fractions_time_series = []

        self.missing_params_not_set = True
        self.random_init()

        if hasattr(self, 'isActivationModel'):
            self.set_activation_functions()

        # record the values at time zero:
        dummy_network = self.params['network'].copy()

        all_nodes_states = list(
            map(lambda node_pointer: 1.0 * self.params['network'].node[node_pointer]['state'],
                self.params['network'].nodes()))
        total_number_of_infected = 2 * np.sum(abs(np.asarray(all_nodes_states)))
        fraction_of_infected = total_number_of_infected / self.params['size']

        if get_time_series:
            network_time_series.append(dummy_network)
            fractions_time_series.append(fraction_of_infected)

        if verbose:
            print('time is', time)
            print('total_number_of_infected is', total_number_of_infected)
            print('total size is', self.params['size'])
        while (total_number_of_infected < cap * self.params['size']) and (not self.spread_stopped):
            self.outer_step()
            dummy_network = self.params['network'].copy()
            time += 1
            all_nodes_states = list(
                map(lambda node_pointer: 1.0 * self.params['network'].node[node_pointer]['state'],
                    self.params['network'].nodes()))
            total_number_of_infected = 2 * np.sum(abs(np.asarray(all_nodes_states)))
            fraction_of_infected = total_number_of_infected / self.params['size']
            if get_time_series:
                network_time_series.append(dummy_network)
                fractions_time_series.append(fraction_of_infected)
            if verbose:
                print('time is', time)
                print('total_number_of_infected is', total_number_of_infected)
                print('total size is', self.params['size'])
            if time > self.params['size'] * 10:
                time = float('Inf')
                print('It is taking too long (10x size) to spread totally.')
                break
        del dummy_network
        if get_time_series:
            return time, total_number_of_infected, network_time_series, fractions_time_series
        else:
            return time, total_number_of_infected

    def generate_network_intervention_dataset(self, dataset_size=200):
        # this can be used for measuring structural properties and how they change as a result of intervention
        interventioned_networks = []
        del interventioned_networks[:]
        for i in range(dataset_size):
            self.missing_params_not_set = True
            self.random_init()
            interventioned_networks += [self.params['network']]

        return interventioned_networks

    def avg_speed_of_spread(self, dataset_size=1000, cap=0.9, mode='max'):
        # avg time to spread over the dataset.
        # The time to spread is measured in one of the modes:
        # integral, max, and total.

        if mode == 'integral':
            integrals = []
            sum_of_integrals = 0
            for i in range(dataset_size):
                _, _, _, infected_fraction_timeseries = self.time_the_total_spread(cap=cap, get_time_series=True)
                integral = sum(infected_fraction_timeseries)
                sum_of_integrals += integral
                integrals += [integral]

            avg_speed = sum_of_integrals / dataset_size
            speed_std = np.std(integrals)
            speed_max = np.max(integrals)
            speed_min = np.min(integrals)
            speed_samples = np.asarray(integrals)

        elif mode == 'max':

            cap_times = []
            sum_of_cap_times = 0
            infection_sizes = []
            sum_of_infection_sizes = 0

            for i in range(dataset_size):
                print('dataset_counter_index is:', i)
                time, infection_size = self.time_the_total_spread(cap=cap, get_time_series=False)
                cap_time = time
                if cap_time == float('Inf'):
                    dataset_size += -1
                    cap_times += [float('Inf')]
                    continue
                sum_of_cap_times += cap_time
                cap_times += [cap_time]
                sum_of_infection_sizes += infection_size
                infection_sizes += [infection_size]

                gc.collect()

            if dataset_size == 0:
                avg_speed = float('Inf')
                speed_std = float('NaN')
                speed_max = float('Inf')
                speed_min = float('Inf')
                speed_samples = np.asarray([float('Inf')])

                avg_infection_size = float('Inf')
                infection_size_std = float('NaN')
                infection_size_max = float('Inf')
                infection_size_min = float('Inf')
                infection_size_samples = np.asarray([float('Inf')])

            else:
                avg_speed = sum_of_cap_times / dataset_size
                speed_std = np.ma.std(cap_times)  # masked entries are ignored
                speed_max = np.max(cap_times)
                speed_min = np.min(cap_times)
                speed_samples = np.asarray(cap_times)

                avg_infection_size = sum_of_infection_sizes / dataset_size
                infection_size_std = np.ma.std(infection_sizes)  # masked entries are ignored
                infection_size_max = np.max(infection_sizes)
                infection_size_min = np.min(infection_sizes)
                infection_size_samples = np.asarray(infection_sizes)

                gc.collect()

        elif mode == 'total':
            total_spread_times = []
            sum_of_total_spread_times = 0
            infection_sizes = []
            sum_of_infection_sizes = 0
            count = 1
            while count <= dataset_size:
                total_spread_time, infection_size = self.time_the_total_spread(cap=0.99999, get_time_series=False)
                if total_spread_time == float('Inf'):
                    dataset_size += -1
                    total_spread_times += [float('Inf')]
                    infection_size += [float('Inf')]
                    print('The contagion hit the time limit.')
                    continue
                total_spread_times += [total_spread_time]
                sum_of_total_spread_times += total_spread_time
                sum_of_infection_sizes += infection_size
                infection_sizes += [infection_size]
                count += 1

            if dataset_size == 0:
                avg_speed = float('Inf')
                speed_std = float('NaN')
                speed_max = float('Inf')
                speed_min = float('Inf')
                speed_samples = np.asarray([float('Inf')])

                avg_infection_size = float('Inf')
                infection_size_std = float('NaN')
                infection_size_max = float('Inf')
                infection_size_min = float('Inf')
                infection_size_samples = np.asarray([float('Inf')])

            else:
                avg_speed = sum_of_total_spread_times / dataset_size
                speed_std = np.std(total_spread_times)
                speed_max = np.max(total_spread_times)
                speed_min = np.min(total_spread_times)
                speed_samples = np.asarray(total_spread_times)

                avg_infection_size = sum_of_infection_sizes / dataset_size
                infection_size_std = np.ma.std(infection_sizes)  # masked entries are ignored
                infection_size_max = np.max(infection_sizes)
                infection_size_min = np.min(infection_sizes)
                infection_size_samples = np.asarray(infection_sizes)

        else:
            assert False, 'undefined mode for avg_speed_of_spread'

        return avg_speed, speed_std, speed_max, speed_min, speed_samples, \
               avg_infection_size, infection_size_std, infection_size_max, infection_size_min, infection_size_samples

    def sample_cost_vs_performance(self, sample_id, cap = 0.9):
        self.missing_params_not_set = True
        node_discovery_cost, edge_discovery_cost = self.random_init()

        time = 0

        if hasattr(self, 'isActivationModel'):
            self.set_activation_functions()

        all_nodes_states = list(
            map(lambda node_pointer: 1.0 * self.params['network'].node[node_pointer]['state'],
                self.params['network'].nodes()))
        total_number_of_infected = 2 * np.sum(abs(np.asarray(all_nodes_states)))

        while (total_number_of_infected < cap * self.params['size']) and (not self.spread_stopped):
            self.outer_step()
            time += 1
            all_nodes_states = list(
                map(lambda node_pointer: 1.0 * self.params['network'].node[node_pointer]['state'],
                    self.params['network'].nodes()))
            total_number_of_infected = 2 * np.sum(abs(np.asarray(all_nodes_states)))
            if time > self.params['size'] * 10:
                time = float('Inf')
                print('It is taking too long (10x size) to spread totally.')
                break

        del(all_nodes_states)
        return total_number_of_infected, node_discovery_cost, edge_discovery_cost

    def get_cost_vs_performance(self, cap=0.9, sample_size = 30, multiprocess = True, num_sample_cpus = 10):
        spread_size_samples = []
        node_discovery_cost_samples = []
        edge_discovery_cost_samples = []

        if not multiprocess:
            for i in range(sample_size):
                spread_size_sample, node_discovery_cost_sample, edge_discovery_cost_sample \
                    = self.sample_cost_vs_performance(sample_id = i, cap = cap)
            
            spread_size_samples.append(spread_size_sample)
            node_discovery_cost_samples.append(node_discovery_cost_sample)
            edge_discovery_cost_samples.append(edge_discovery_cost_sample)
        
        else:
            partial_sample_cost_vs_performance = partial(self.sample_cost_vs_performance,
                                                         cap = cap)
            with Multipool(processes = num_sample_cpus) as pool:
                spread_data = pool.map(partial_sample_cost_vs_performance, list(range(sample_size)))

            spread_size_samples = [spread_data[i][0] for i in range(sample_size)]
            node_discovery_cost_samples = [spread_data[i][1] for i in range(sample_size)]
            edge_discovery_cost_samples = [spread_data[i][2] for i in range(sample_size)]

        if node_discovery_cost_samples[0] is None:
            return (np.average(spread_size_samples), np.std(spread_size_samples), sum(spread < 10.0 for spread in spread_size_samples))
        else:
            return (np.average(spread_size_samples), np.std(spread_size_samples), sum(spread < 10.0 for spread in spread_size_samples),
                    np.average(node_discovery_cost_samples), np.std(node_discovery_cost_samples),
                    np.average(edge_discovery_cost_samples), np.std(edge_discovery_cost_samples))

    def outer_step(self):
        assert hasattr(self, 'classification_label'), 'classification_label not set'
        assert not self.missing_params_not_set, 'missing params are not set'
        self.number_of_active_infected_neighbors_is_updated = False
        self.time_since_infection_is_updated = False
        self.time_since_activation_is_updated = False

        self.step()

        gc.collect()

        assert self.time_since_infection_is_updated \
               and self.time_since_activation_is_updated \
               and self.number_of_active_infected_neighbors_is_updated \
               and self.list_of_inactive_infected_agents_is_updated \
               and self.list_of_active_infected_agents_is_updated \
               and self.list_of_susceptible_agents_is_updated \
               and self.list_of_exposed_agents_is_updated \
               and self.list_of_most_recent_activations_is_updated, \
            "error states or list mishandled"

    def step(self):
        # implement this in class children
        pass

    def seed(self):
        # implement this in class children
        pass


class IndependentCascade(ContagionModel):
    """
    Implements an independent cascade model. Each infected neighbor has an independent probability beta of passing on her
    infection, as long as her infection has occurred within the past mem = 1 time steps.
    For plain vanila IC we should have:
    self.params['delta'] = 0,
    self.params['alpha'] = 0,
    self.params['gamma'] = 1,
    self.params['memory'] = 0
    """

    def __init__(self, params):
        super(IndependentCascade, self).__init__(params)
        self.classification_label = SIMPLE

        assert TRACK_TIME_SINCE_VARIABLES, "we need the time_since_variables for the IndependentCascade model"

    def step(self):

        SOMETHING_HAPPENED = False

        current_network = copy.deepcopy(self.params['network'])

        for i in current_network.nodes():
            # current_network.node[i]['state'] can either be susceptible (0)
            # or active infected (0.5) or inactive infected (-0.5)

            # transition from susceptible to active infected:
            if current_network.node[i]['state'] == susceptible:
                assert self.params['network'].node[i]['time_since_infection'] == 0 and \
                       self.params['network'].node[i]['time_since_activation'] == 0, \
                    'error: time_since_infection or time_since_activation mishandled!'
                self.time_since_infection_is_updated = True
                self.time_since_activation_is_updated = True

                for j in current_network.neighbors(i):
                    if current_network.node[j]['state'] == infected * active:
                        assert current_network.node[j]['time_since_activation'] <= self.params['memory'], \
                            "error: should not remain active beyond mem times."
                        if RD.random() < self.params['beta']:

                            self.params['network'].node[i]['state'] = infected * active
                            SOMETHING_HAPPENED = True

                            for k in self.params['network'].neighbors(i):
                                self.params['network'].node[k]['number_of_active_infected_neighbors'] += 1
                            break

                self.number_of_active_infected_neighbors_is_updated = True

                # in all the below cases the node is in infected active or infected inactive state.

            # transition from active or inactive infected to susceptible:
            elif RD.random() < self.params['delta']:
                assert 2 * abs(current_network.node[i]['state']) == infected, \
                    "error: node states are mishandled"
                #  here the node should either be active infected (+0.5) or inactive infected (-0.5)

                assert self.params['network'].node[i]['time_since_activation'] <= self.params['memory'], \
                    "error: time_since_activation should not get greater than mem"
                self.params['network'].node[i]['state'] = susceptible

                if current_network.node[i]['state'] == infected * active:
                    for k in self.params['network'].neighbors(i):
                        assert self.params['network'].node[k]['number_of_active_infected_neighbors'] > 0, \
                            'error: number_of_active_infected_neighbors is mishandled'
                        # here number_of_active_infected_neighbors for neighbor k should be at least one
                        self.params['network'].node[k]['number_of_active_infected_neighbors'] -= 1
                self.number_of_active_infected_neighbors_is_updated = True
                self.params['network'].node[i]['time_since_infection'] = 0
                self.params['network'].node[i]['time_since_activation'] = 0
                self.time_since_infection_is_updated = True
                self.time_since_activation_is_updated = True

            # transition from active infected to inactive infected:

            elif ((current_network.node[i]['state'] == infected * active
                   and (RD.random() < self.params['gamma'])) and
                  current_network.node[i]['time_since_activation'] == self.params['memory']):
                self.params['network'].node[i]['state'] = infected * inactive
                for k in self.params['network'].neighbors(i):
                    assert self.params['network'].node[k]['number_of_active_infected_neighbors'] > 0, \
                        'error: number_of_active_infected_neighbors is mishandled'
                    # here number_of_active_infected_neighbors for neighbor k should be at least one
                    self.params['network'].node[k]['number_of_active_infected_neighbors'] -= 1
                self.number_of_active_infected_neighbors_is_updated = True
                self.params['network'].node[i]['time_since_infection'] += 1
                self.params['network'].node[i]['time_since_activation'] = 0
                self.time_since_infection_is_updated = True
                self.time_since_activation_is_updated = True

            # transition from inactive infected to active infected:

            elif current_network.node[i]['state'] == infected * inactive and RD.random() < self.params['alpha']:
                assert self.params['network'].node[i]['time_since_activation'] == 0, \
                    "error: time_since_activation should be zero for an inactive node"
                self.params['network'].node[i]['state'] = infected * active
                for k in self.params['network'].neighbors(i):
                    self.params['network'].node[k]['number_of_active_infected_neighbors'] += 1
                self.number_of_active_infected_neighbors_is_updated = True
                self.params['network'].node[i]['time_since_infection'] += 1
                self.params['network'].node[i]['time_since_activation'] = 0
                self.time_since_infection_is_updated = True
                self.time_since_activation_is_updated = True

            # else the node state is either active or inactive infected and
            # there are no state transitions, but we still need to update the time_since variables:

            else:
                assert self.params['network'].node[i]['time_since_activation'] <= self.params['memory'], \
                    "error: time_since_activation should be less than or equal to mem"
                if current_network.node[i]['state'] == infected * inactive:
                    assert self.params['network'].node[i]['time_since_activation'] == 0, \
                        "error: time_since_activation should be zero for an inactive node"
                self.params['network'].node[i]['time_since_infection'] += 1
                if current_network.node[i]['state'] == infected * active:
                    self.params['network'].node[i]['time_since_activation'] += 1
                self.time_since_infection_is_updated = True
                self.time_since_activation_is_updated = True
                self.number_of_active_infected_neighbors_is_updated = True

        del current_network

        if (self.params['delta'] == 0) and (self.params['alpha'] == 0):  # plain vanila IC
            if not SOMETHING_HAPPENED:
                self.spread_stopped = True


class IndependentCascadeRandomSeeding(IndependentCascade):

    def __init__(self, params):
        super(IndependentCascadeRandomSeeding, self).__init__(params)
        self.classification_label = SIMPLE  # or COMPLEX
        # set other model specific flags and handles here

    def query(self):
        """
        queries the graph structure
        """

        assert self.network_initialized, "cannot query the network structure if it is not initialized"
        assert not self.states_initialized, "states cannot be initialized before queries"

        if 'k' in self.fixed_params:
            self.params['k'] = self.fixed_params['k']
        else:
            self.params['k'] = 2
            print('warning: k was not provided, set to 2')

        k_random_node_indices = np.random.choice(list(self.params['network'].nodes()),
                                                 self.params['k'],
                                                 replace=False)

        return k_random_node_indices

    def seed(self):
        k_random_node_indices = self.query()
        return k_random_node_indices


class IndependentCascadeEdgeQuerySeeding(IndependentCascade):

    def __init__(self, params):
        super(IndependentCascadeEdgeQuerySeeding, self).__init__(params)

        #Question: initiate probe parameters here, calculate them from epsilon?

    def reveal_nbhr(self, node):
        """
        Reveal a neighbor of the given node
        """

        assert self.network_initialized, "cannot query the network structure if it is not initialized"
        assert not self.states_initialized, "states cannot be initialized before queries"

        reveal_neighbor = np.random.choice([True, False], 
                                           1, 
                                           False, 
                                           [self.params['beta'], 1 - self.params['beta']])

        return np.random.choice(list(self.params['network'].neighbors(node)), 1)[0] if reveal_neighbor else None

    def probe(self, sampled_nodes):
        """
        probes the graph structure for edge query seeding
        """
        # Assumption: probe parameters rho, T and tau are assumed to be provided on init

        probed_subgraphs = []
        discovered_nodes = set()
        discovered_edges = set()

        for i in range(self.params['T']):
            probed_subgraph = []

            candidate_nodes = set(sampled_nodes)
            predecessor_table = {candidate_node : None for candidate_node in candidate_nodes}

            while len(candidate_nodes) != 0:
                candidate_node = candidate_nodes.pop()

                discovered_nodes.add(candidate_node)
                next_candidate_node = self.reveal_nbhr(candidate_node)

                if next_candidate_node is not None:
                    candidate_nodes.add(next_candidate_node)
                    predecessor_table[next_candidate_node] = candidate_node

                    if (candidate_node, next_candidate_node) not in discovered_edges \
                        and (next_candidate_node, candidate_node) not in discovered_edges:
                        discovered_edges.add((candidate_node, next_candidate_node))

                in_new_component = True
                for component in probed_subgraph:
                    if predecessor_table[candidate_node] in component:
                        component.add(candidate_node)
                        in_new_component = False
                        if len(component) > self.params['tau']:
                            candidate_nodes.difference_update(component)

                if in_new_component:
                    probed_subgraph.append({candidate_node})

            probed_subgraphs.append(probed_subgraph)
            del(candidate_nodes)
            del(predecessor_table)

        node_discovery_cost = len(discovered_nodes)
        edge_discovery_cost = len(discovered_edges)
        del(sampled_nodes)
        del(discovered_nodes)
        del(discovered_edges)
        return probed_subgraphs, node_discovery_cost, edge_discovery_cost

    def query(self):
        sample_size = int(self.params['size'] * self.params['rho'])
        sampled_nodes = np.random.choice(list(self.params['network'].nodes()),
                                         sample_size,
                                         replace = False)
        probed_subgraphs, node_discovery_cost, edge_discovery_cost = self.probe(sampled_nodes)

        all_components = list(itertools.chain.from_iterable(probed_subgraphs))
        sampled_nodes_set = set(sampled_nodes)
        component_scores = [len(component.intersection(sampled_nodes_set)) for component in all_components]

        del(sample_size)
        del(sampled_nodes)
        del(probed_subgraphs)
        del(sampled_nodes_set)
        return (all_components, component_scores, node_discovery_cost, edge_discovery_cost)
    
    def seed(self):
        all_components, component_scores, node_discovery_cost, edge_discovery_cost = self.query()

        candidate_sample_size = int((self.params['size'] / self.params['k']) * np.log(1 / self.params['eps_prime']))
        search_set = set(self.params['network'].nodes())
        seed_set = []

        for i in range(self.params['k']):
            candidate_sample = np.random.choice(list(search_set), 
                                                size = candidate_sample_size,
                                                replace = False)

            next_seed = None
            max_gain = -1
            for candidate in candidate_sample:
                candidate_score = 0
                for i in range(len(all_components)):
                    if candidate in all_components[i]:
                        candidate_score += component_scores[i]

                if candidate_score > max_gain:
                    next_seed = candidate
                    max_gain = candidate_score

            if next_seed is not None:
                seed_set.append(next_seed)
                search_set.remove(next_seed)
                for i in range(len(all_components)):
                    if next_seed in all_components[i]:
                        component_scores[i] = 0

        del(all_components)
        del(component_scores)
        del(candidate_sample_size)
        del(search_set)

        return (seed_set, node_discovery_cost, edge_discovery_cost)


class IndependentCascadeSpreadQuerySeeding(IndependentCascade):

    def __init__(self, params):
        super(IndependentCascadeSpreadQuerySeeding, self).__init__(params)

    def spread(self, i):
        dummy_contagion_model = IndependentCascade(copy.deepcopy(self.params))

        dummy_contagion_model.missing_params_not_set = True
        dummy_contagion_model.random_init(with_seed = False)

        dummy_contagion_model.init_network_states([i])

        if hasattr(self, 'isActivationModel'):
            dummy_contagion_model.set_activation_functions()

        all_nodes_states = list(
            map(lambda node_pointer: 1.0 * dummy_contagion_model.params['network'].node[node_pointer]['state'],
                dummy_contagion_model.params['network'].nodes()))
        total_number_of_infected = 2 * np.sum(abs(np.asarray(all_nodes_states)))

        time = 0
        # Note: the assumption here is that the tau paramter indicates the cap of maximum spread, similar to edge query model
        while (total_number_of_infected < dummy_contagion_model.params['tau']) \
            and (not dummy_contagion_model.spread_stopped):
            time += 1
            dummy_contagion_model.outer_step()

            all_nodes_states = list(
                map(lambda node_pointer: 1.0 * dummy_contagion_model.params['network'].node[node_pointer]['state'],
                    dummy_contagion_model.params['network'].nodes()))
            total_number_of_infected = 2 * np.sum(abs(np.asarray(all_nodes_states)))
            
            if time > self.params['size'] * 10:
                break 

        spread = set(filter(lambda j : dummy_contagion_model.params['network'].node[j]['state'] != 0,
                            dummy_contagion_model.params['network'].nodes()))
        del(dummy_contagion_model)
        return spread

    def query(self):
        sampled_spreads = []
        sampled_nodes = np.random.choice(list(self.params['network'].nodes()),
                                         size = int(self.params['rho']),
                                         replace = False)
        
        for node in sampled_nodes:
            sampled_spreads.append(self.spread(node))

        return sampled_spreads

    def seed(self):
        candidates = set(list(self.params['network'].nodes()))
        seeds = []

        for i in range(self.params['k']):
            sampled_spreads = self.query()

            for j in range(len(sampled_spreads)):
                if sampled_spreads[j].intersection(set(seeds)):
                    sampled_spreads[j] = set()

            new_seed = None
            max_score = 0
            for candidate in candidates:
                candidate_score = np.sum([candidate in spread for spread in sampled_spreads])
                if candidate_score >= max_score:
                    new_seed = candidate
                    max_score = candidate_score

            if new_seed is not None:
                seeds.append(new_seed)
                candidates.remove(new_seed)
            del(sampled_spreads)

        del(candidates)
        return (seeds, None, None)


class IndependentCascadeGreedySeeding(IndependentCascade):

    def __init__(self, params):
        super(IndependentCascadeGreedySeeding, self).__init__(params)

    def spread(self, initially_infected):
        dummy_contagion_model = IndependentCascade(copy.deepcopy(self.params))

        dummy_contagion_model.missing_params_not_set = True
        dummy_contagion_model.random_init(with_seed = False)

        dummy_contagion_model.init_network_states(initially_infected)

        if hasattr(self, 'isActivationModel'):
            dummy_contagion_model.set_activation_functions()

        all_nodes_states = list(
            map(lambda node_pointer: 1.0 * dummy_contagion_model.params['network'].node[node_pointer]['state'],
                dummy_contagion_model.params['network'].nodes()))
        total_number_of_infected = 2 * np.sum(abs(np.asarray(all_nodes_states)))

        time = 0
        # Note: the assumption here is that the tau paramter indicates the cap of maximum spread, similar to edge query model
        while (total_number_of_infected < dummy_contagion_model.params['tau']) \
            and (not dummy_contagion_model.spread_stopped):
            time += 1
            dummy_contagion_model.outer_step()

            all_nodes_states = list(
                map(lambda node_pointer: 1.0 * dummy_contagion_model.params['network'].node[node_pointer]['state'],
                    dummy_contagion_model.params['network'].nodes()))
            total_number_of_infected = 2 * np.sum(abs(np.asarray(all_nodes_states)))
            
            if time > self.params['size'] * 10:
                break 

        spread = set(filter(lambda j : dummy_contagion_model.params['network'].node[j]['state'] != 0,
                            dummy_contagion_model.params['network'].nodes()))
        del(dummy_contagion_model)
        return spread

    def sample_spread(self, initially_infected, sample_size):
        sample_seeds = [initially_infected for sample_id in range(sample_size)]

        if self.params['multiprocess_mg_sample']:
            with Multipool(processes = self.params['num_mg_sample_cpus']) as pool:
                sample_spreads = pool.map(self.spread, sample_seeds)
        else:
            sample_spreads = []
            for initially_infected in sample_seeds:
                sample_spreads.append(self.spread(initially_infected))

        return np.average(sample_spreads)

    def perform_next_iter(self, seeds, queue):
        _, iter_flag, node = heapq.heappop(queue)

        if iter_flag == len(seeds):
            seeds.append(node)
            print('seed', len(seeds), 'added')
        else:
            seeds_with_extra_node = copy.deepcopy(seeds) + [node]
            new_negated_marginal_gain = (self.sample_spread(seeds, self.params['mg_sample_size'])
                                        - self.sample_spread(seeds_with_extra_node, self.params['mg_sample_size']))
            new_iter_flag = len(seeds)
            heapq.heappush(queue, (new_negated_marginal_gain, new_iter_flag, node))

    def seed(self):
        queue = [(float('-inf'), -1, node) for node in self.params['network'].nodes()]
        heapq.heapify(queue)
        seeds = []

        while len(seeds) < self.params['k']:
            self.perform_next_iter(seeds, queue)

        return (seeds, None, None)
