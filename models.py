from settings import *
import gc
import copy
import itertools
import heapq

from functools import partial

from Multipool import Multipool

TRACK_TIME_SINCE_VARIABLES = True


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
            pass

        # when considering real network and interventions on them we may need to record the original network.
        # This is currently only used in SimpleOnlyAlongOriginalEdges(ContagionModel)

        if 'original_network' in self.fixed_params:
            self.params['original_network'] = self.fixed_params['original_network']
        else:
            self.params['original_network'] = None

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
            seed_set, node_discovery_cost, secondary_cost = self.seed()
        else:
            seed_set = []

        self.init_network_states(seed_set)

        self.missing_params_not_set = False

        self.spread_stopped = False

        if with_seed:
            return node_discovery_cost, secondary_cost

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
        node_discovery_cost, secondary_cost = self.random_init()

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
        return total_number_of_infected, node_discovery_cost, secondary_cost

    def get_cost_vs_performance(self, cap=0.9, sample_size = 30, multiprocess = True, num_sample_cpus = 10):
        spread_size_samples = []
        node_discovery_cost_samples = []
        secondary_cost_samples = []

        if not multiprocess:
            for i in range(sample_size):
                spread_size_sample, node_discovery_cost_sample, secondary_cost_sample \
                    = self.sample_cost_vs_performance(sample_id = i, cap = cap)
            
            spread_size_samples.append(spread_size_sample)
            node_discovery_cost_samples.append(node_discovery_cost_sample)
            secondary_cost_samples.append(secondary_cost_sample)
        
        else:
            partial_sample_cost_vs_performance = partial(self.sample_cost_vs_performance,
                                                         cap = cap)
            with Multipool(processes = num_sample_cpus) as pool:
                spread_data = pool.map(partial_sample_cost_vs_performance, list(range(sample_size)))

            spread_size_samples = [spread_data[i][0] for i in range(sample_size)]
            node_discovery_cost_samples = [spread_data[i][1] for i in range(sample_size)]
            secondary_cost_samples = [spread_data[i][2] for i in range(sample_size)]

        if node_discovery_cost_samples[0] is None:
            return (np.average(spread_size_samples), np.std(spread_size_samples), sum(spread < 10.0 for spread in spread_size_samples))
        else:
            return (np.average(spread_size_samples), np.std(spread_size_samples), sum(spread < 10.0 for spread in spread_size_samples),
                    np.average(node_discovery_cost_samples), np.std(node_discovery_cost_samples),
                    np.average(secondary_cost_samples), np.std(secondary_cost_samples))
                    
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
        return k_random_node_indices, None, None


class IndependentCascadeNominationSeeding(IndependentCascade):

    def __init__(self, params):
        super(IndependentCascadeNominationSeeding, self).__init__(params)
        self.classification_label = SIMPLE

    def query(self):
        pass

    def seed(self):
        pass
        

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
        unique_nodes_discovered = set()
        sum_of_spreads = 0

        for i in range(self.params['k']):
            sampled_spreads = self.query()

            for spread in sampled_spreads:
                unique_nodes_discovered.update(spread)
                sum_of_spreads += len(spread)

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
        return (seeds, len(unique_nodes_discovered), sum_of_spreads)


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

        del(dummy_contagion_model)
        return total_number_of_infected

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
