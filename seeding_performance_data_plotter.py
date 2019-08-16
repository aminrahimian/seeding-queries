import os
import numpy as np
import matplotlib.pyplot as plt
import pickle

colors = {2 : 'red', 4 : 'orange', 8 : 'blue', 10 : 'black'}


def dump_seeding_performance_plot(data_type, seeding_model, external_list_of_k = None, logscaled = True, xticks = None):
    root_folder = 'data/' + data_type

    data_folder = root_folder + '/pickled_samples/spreading_pickled_samples/' + seeding_model +'/'
    filenames = os.listdir(data_folder)

    plot_folder = root_folder + '/plots/'
    os.makedirs(os.path.dirname(plot_folder), exist_ok = True)

    data = {}
    list_of_k = set()

    for name in filenames:
        name_bits = name[:-4].split('_')
        
        k = int(name_bits[4])
        list_of_k.add(k)
        if k not in data:
            data[k] = {}    

        if seeding_model == 'spread_query':
            q = float(name_bits[8])
            s = int(name_bits[11])
        else:
            q = float(name_bits[6])
            s = int(name_bits[9])
        
        data[k][q] = (pickle.load(open(data_folder + name, 'rb')), s)

    list_of_k = sorted(list_of_k)
    if external_list_of_k is not None:
        list_of_k = external_list_of_k

    fig = plt.figure()
    fig.set_figwidth(7)
    fig.set_figheight(5)

    for k in list_of_k:
        x = sorted(data[k].keys())

        y = []
        iv = []
        
        for q in x:
            y.append(data[k][q][0][0])
            iv.append(1.96 * data[k][q][0][1] / (data[k][q][1]**0.5))
        
        plt.errorbar(x, y, iv, color = colors[k], label = 'k=' + str(k))

    plt.xlabel('Query cost')
    if logscaled:
        plt.xscale('log')
    if xticks is not None:
        plt.xticks(xticks, xticks)
    
    plt.ylabel('Infection size')  
    plt.legend(fontsize = 'large')

    fig.savefig(plot_folder + seeding_model + '_seeding_plot_' + 'k_' + str(list_of_k) + '.pdf',
                bbox_inches = 'tight')
    plt.close()


if __name__ == "__main__":
    data_type = 'fb100-data'
    seeding_model = 'edge_query'
    external_list_of_k = None # [8, 10]

    logscaled = False
    xticks = None #[8, 16, 32, 72, 140, 1000]

    dump_seeding_performance_plot(data_type, seeding_model, external_list_of_k, logscaled, xticks)
