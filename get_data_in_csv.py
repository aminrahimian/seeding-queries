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
        writer.writerow(['k', 'T', 'seed set id', 'sample id', 'spread size', 
                         'edge cost', 'edge cost without leaves', 
                         'node cost', 'node cost without leaves'])

        for k in [2, 4, 10]:
            random_spread = pickle.load(open(root + 'k_' + str(k)
                                              + '/edge_query'
                                              + '/' + network_id
                                              + '/random_spread_size_samples_'
                                              + network_group + network_id
                                              + model_id + '.pkl', 'rb'))
            random_cost = [(0, 0) for i in range(50)]

            for seed_set_id in range(50):
                for sample_id in range(500):
                    writer.writerow([k, 0, seed_set_id, sample_id, 
                                    random_spread[seed_set_id][sample_id],
                                    random_cost[seed_set_id][0], random_cost[seed_set_id][0],
                                    random_cost[seed_set_id][1], random_cost[seed_set_id][1]])
            

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
                costs_without_leaves = pickle.load(open(root + 'k_' + str(k)
                                                        + '/edge_query'
                                                        + '/' + network_id
                                                        + '/cost_samples_without_leaf_'
                                                        + network_group + network_id
                                                        + '_T_' + str(T)
                                                        + model_id + '.pkl', 'rb'))

                for seed_set_id in range(50):
                    for sample_id in range(500):
                        writer.writerow([k, T, seed_set_id, sample_id, 
                                        spreads[50 * seed_set_id + sample_id],
                                        costs[seed_set_id][0], costs_without_leaves[seed_set_id][0],
                                        costs[seed_set_id][1], costs_without_leaves[seed_set_id][1]])


if __name__ == '__main__':
    get_edge_query_data()