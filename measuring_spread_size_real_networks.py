# compute the spread time in the original network and under edge addition and rewiring interventions

from models import *

from pathlib import Path


VERBOSE = True

CHECK_FOR_EXISTING_PKL_SAMPLES = False

size_of_dataset = 10

CAP = 0.9


def measure_spread_size(network_id):
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

    if CHECK_FOR_EXISTING_PKL_SAMPLES:
        path = Path(spreading_pickled_samples_directory_address + 'infection_size_original_'
                    + network_group + network_id
                    + model_id + '.pkl')
        if path.is_file():
            print('infection_size_original_'
                  + network_group + network_id
                  + model_id + ' already exists')
            return

    params_original = {
        'network': G,
        'original_network': G,
        'size': network_size,
        'add_edges': False,
        'k': k,
        'delta': delta,
        'alpha': alpha,
        'beta': beta,
        'gamma': gamma,
        'memory': memory,
        'rewire': False,
        'rewiring_mode': 'random_random',
        'num_edges_for_random_random_rewiring': None,
    }
    if model_id == '_vanilla IC_':
        dynamics = IndependentCascadeRandomSeeding(params_original)
    else:
        print('model_id is not valid')
        exit()
    speed_original, std_original, _, _, speed_samples_original, \
        infection_size_original, infection_size_std_original, _, _, infection_size_samples_original = \
        dynamics.avg_speed_of_spread(
            dataset_size=size_of_dataset,
            cap=CAP,
            mode='max')
    if VERBOSE:
        print('mean time to spread:', speed_original, std_original)
        print('mean infection_size:', infection_size_original, infection_size_std_original)
        print('spread time samples:', speed_samples_original)
        print('infection size samples:', infection_size_samples_original)

    if save_computations:

        pickle.dump(speed_samples_original, open(spreading_pickled_samples_directory_address
                                                 + 'speed_samples_original_'
                                                 + network_group + network_id
                                                 + model_id + '.pkl', 'wb'))

        pickle.dump(infection_size_samples_original, open(spreading_pickled_samples_directory_address
                                                          + 'infection_size_original_'
                                                          + network_group + network_id
                                                          + model_id + '.pkl', 'wb'))


if __name__ == '__main__':

    assert do_computations, "we should be in do_computations mode"

    if do_multiprocessing:
        with multiprocessing.Pool(processes=number_CPU) as pool:

            pool.map(measure_spread_size, network_id_list)

    else:  # no multi-processing
        # do computations for the original networks:

        for network_id in network_id_list:
            measure_spread_size(network_id)
