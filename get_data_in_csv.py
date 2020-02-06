from settings import *
import os
import csv
import pickle
import numpy as np

root = './data/fb100-data/pickled_samples/spreading_pickled_samples/'
costs = [0, 1, 2, 3, 4, 5, 7, 8, 10, 13, 16, 20, 24, 30, 37, 46, 57]
spread_costs_k_2_and_4 = [0, 4, 8, 12, 16, 20, 24, 32, 44, 60, 80, 104, 132, 164, 200]
spread_costs_k_10 = [0, 10, 20, 30, 40, 50, 60, 80, 100, 120, 140, 160, 180, 200]

def get_edge_query_data():
    filename = root + 'edge_query_rho_100' + '.csv'
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, 'w') as f:
        writer = csv.writer(f, delimiter = ',')
        writer.writerow(['k', 'T', 'seed set id', 'sample id', 'spread size', 
                         'edge cost', 'edge cost without leaves', 
                         'node cost', 'node cost without leaves'])

        for k in [2, 4, 10]:       
            for T in costs:
                spreads = pickle.load(open(root + 'k_' + str(k)
                                        + '/edge_query'
                                        + '/' + network_id
                                        + '/spread_size_samples_'
                                        + network_group + network_id
                                        + '_T_' + str(T)
                                        + model_id + '.pkl', 'rb'))
                
                cost_samples = pickle.load(open(root + 'k_' + str(k)
                                        + '/edge_query'
                                        + '/' + network_id
                                        + '/cost_samples_'
                                        + network_group + network_id
                                        + '_T_' + str(T)
                                        + model_id + '.pkl', 'rb'))

                for seed_set_id in range(50):
                    for sample_id in range(500):
                        writer.writerow([k, T, seed_set_id, sample_id, 
                                        spreads[500 * seed_set_id + sample_id],
                                        cost_samples[seed_set_id][0], cost_samples[seed_set_id][1],
                                        cost_samples[seed_set_id][2], cost_samples[seed_set_id][3]])


def get_spread_query_data():
    filename = root + 'spread_query' + '.csv'
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, 'w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['k', 'query cost', 'seed set id', 'sample id', 'spread size',
                         'edge cost', 'edge cost without leaves',
                         'node cost', 'node cost without leaves'])

        for k in [2, 4]:
            for query_cost in spread_costs_k_2_and_4:
                spreads = pickle.load(open(root + 'k_' + str(k)
                                           + '/spread_query'
                                           + '/' + network_id + '/'
                                           + 'spread_size_samples_'
                                           + network_group + network_id
                                           + '_query_cost_' + str(query_cost)
                                           + model_id + '.pkl', 'rb'))

                cost_samples = pickle.load(open(root + 'k_' + str(k)
                                                + '/spread_query'
                                                + '/' + network_id + '/'
                                                + 'cost_samples_'
                                                + network_group + network_id
                                                + '_query_cost_' + str(query_cost)
                                                + model_id + '.pkl', 'rb'))

                for seed_set_id in range(50):
                    for sample_id in range(500):
                        writer.writerow([k, query_cost, seed_set_id, sample_id,
                                         spreads[500 * seed_set_id + sample_id],
                                         cost_samples[seed_set_id][0], cost_samples[seed_set_id][1],
                                         cost_samples[seed_set_id][2], cost_samples[seed_set_id][3]])
        for k in [10]:
            for query_cost in spread_costs_k_10:
                spreads = pickle.load(open(root + 'k_' + str(k)
                                           + '/spread_query'
                                           + '/' + network_id + '/'
                                           + 'spread_size_samples_'
                                           + network_group + network_id
                                           + '_query_cost_' + str(query_cost)
                                           + model_id + '.pkl', 'rb'))

                cost_samples = pickle.load(open(root + 'k_' + str(k)
                                                + '/spread_query'
                                                + '/' + network_id + '/'
                                                + 'cost_samples_'
                                                + network_group + network_id
                                                + '_query_cost_' + str(query_cost)
                                                + model_id + '.pkl', 'rb'))

                for seed_set_id in range(50):
                    for sample_id in range(500):
                        writer.writerow([k, query_cost, seed_set_id, sample_id,
                                         spreads[500 * seed_set_id + sample_id],
                                         cost_samples[seed_set_id][0], cost_samples[seed_set_id][1],
                                         cost_samples[seed_set_id][2], cost_samples[seed_set_id][3]])


if __name__ == '__main__':
    # get_edge_query_data()
    get_spread_query_data()
