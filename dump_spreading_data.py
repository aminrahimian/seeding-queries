# goes over pickled spreading times  and puts their into the CSV file: network_group + 'spreading_data_dump.csv'

from settings import *

include_spread_size = True

if not include_spread_size:

    tracked_properties = ['network_group',
                          'network_id',
                          'network_size',
                          'intervention_type',
                          'intervention_size',
                          'sample_id',
                          'time_to_spread',
                          'model']
else:

    tracked_properties = ['network_group',
                          'network_id',
                          'network_size',
                          'intervention_type',
                          'intervention_size',
                          'sample_id',
                          'time_to_spread',
                          'model',
                          'size_of_spread']


assert include_spread_size, "data dump without spread size is not supported!"

update_existing_dump = False


if __name__ == "__main__":

    assert data_dump, "we should be in data_dump mode!"

    assert load_computations, "we should be in load_computations mode to dump data!"

    # check for the existing network_group + 'spreading_data_dump.csv' file

    generating_new_dump = False

    try:
        df = pd.read_csv(output_directory_address + network_group + 'spreading_data_dump.csv')
        print('read_csv', df)
    except IOError:

        df = pd.DataFrame(columns=tracked_properties, dtype='float')
        print('New ' + network_group + 'spreading_data_dump file will be generated.')
        generating_new_dump = True

    if update_existing_dump:
        assert not generating_new_dump, "we should not be generating a new spreading_data_dump file."

    for network_id in network_id_list:

        print('load/dump speed_samples_original_' + network_group + network_id + model_id)

        #  load in the network and extract preliminary data

        fh = open(edgelist_directory_address + network_group + network_id + '.txt', 'rb')

        G = NX.read_edgelist(fh, delimiter=DELIMITER)

        print('original size ', len(G.nodes()))

        #  get the largest connected component:
        if not NX.is_connected(G):
            G = max(NX.connected_component_subgraphs(G), key=len)
            print('largest connected component extracted with size ', len(G.nodes()))

        network_size = NX.number_of_nodes(G)



        speed_samples_original = pickle.load(open(spreading_pickled_samples_directory_address
                                                  + 'speed_samples_original_'
                                                  + network_group + network_id
                                                  + model_id + '.pkl', 'rb'))

        spread_samples_original = pickle.load(open(spreading_pickled_samples_directory_address
                                                   + 'infection_size_original_'
                                                   + network_group + network_id
                                                   + model_id + '.pkl', 'rb'))

        speed_original = np.mean(speed_samples_original)

        spread_original = np.mean(spread_samples_original)

        std_original = np.std(speed_samples_original)

        spread_std_original = np.std(spread_samples_original)

        # dump original:

        df_common_part_original = pd.DataFrame(data=[[network_group, network_id, network_size,
                                                      'none', 0.0, MODEL]] * len(speed_samples_original),
                                               columns=['network_group',
                                                        'network_id',
                                                        'network_size',
                                                        'intervention_type',
                                                        'intervention_size',
                                                        'model'])

        df_sample_ids_original = pd.Series(list(range(len(speed_samples_original))), name='sample_id')

        df_time_to_spreads_original = pd.Series(speed_samples_original, name='time_to_spread')

        df_size_of_spreads_original = pd.Series(spread_samples_original, name='size_of_spread')

        new_df_original = pd.concat([df_common_part_original,
                                     df_sample_ids_original,
                                     df_time_to_spreads_original,
                                     df_size_of_spreads_original],
                                    axis=1)

        print(new_df_original)

        extended_frame = [df, new_df_original]

        df = pd.concat(extended_frame, ignore_index=True, verify_integrity=False).drop_duplicates().reset_index(
            drop=True)

        print(df)

        df.to_csv(output_directory_address + network_group + 'spreading_data_dump.csv', index=False)  # , index=False
