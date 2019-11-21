from settings import *
import os
import csv
import pickle
import numpy as numpy

root = './data/fb100-data/pickled_samples/spreading_pickled_samples/'


def get_edge_query_data():
    filename = root + 'edge_query' + '.csv'
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, 'w') as f:
        writer = csv.writer(f, delimiter = ',')
        writer.writerow(['k', 'T', 'seed set id', 'sample id', 'spread size', 'edge cost', 'node cost'])

        for k in [2, 4, 10]:
            for T in range(1,16):
                spreads = pickle.load(open(root + 'k_' + str(k)
                                        + '/edge_query'
                                        + '/' + network_id
                                        + '/spread_size_samples_'
                                        + network_group + network_id
                                        + '_T_' + str(T)
                                        + model_id + '.pkl', 'rb'))
                costs = pickle.load(open(root + 'k_' + str(k)
                                        + '/edge_query'
                                        + '/' + network_id
                                        + '/cost_samples_'
                                        + network_group + network_id
                                        + '_T_' + str(T)
                                        + model_id + '.pkl', 'rb'))

                for seed_set_id in range(50):
                    for sample_id in range(500):
                        writer.writerow([k, T, seed_set_id, sample_id, 
                                        spreads[50 * seed_set_id + sample_id],
                                        costs[seed_set_id][0], costs[seed_set_id][1]])


if __name__ == '__main__':
    get_edge_query_data()