import pickle
import os


def save_data(errors, network_params, time_array, save_dir, suffix):
    # Save these into a pickle file
    [local_params, nbrs_info, misc]                    = network_params
    (mean_MT_SE, mean_data_SE, all_MT_SE)              = errors
    (Q, T, S_collection, P, Xi)                        = local_params
    (N1, W1, N0, W0)                                   = nbrs_info
    (tangent_colors, D, d, M, P)                       = misc

    os.makedirs(save_dir, exist_ok=True)

    # Save each group in a separate pickle file
    with open(os.path.join(save_dir, f'errors_{suffix}.pkl'), 'wb') as f:
        pickle.dump((mean_MT_SE, mean_data_SE, all_MT_SE), f)

    with open(os.path.join(save_dir, f'local_params_{suffix}.pkl'), 'wb') as f:
        pickle.dump((Q, T, S_collection, P, Xi), f)

    with open(os.path.join(save_dir, f'nbrs_info_{suffix}.pkl'), 'wb') as f:
        pickle.dump((N1, W1, N0, W0), f)

    with open(os.path.join(save_dir, f'time_array_{suffix}.pkl'), 'wb') as f:
        pickle.dump(time_array, f)

    with open(os.path.join(save_dir, f'misc_{suffix}.pkl'), 'wb') as f:
        pickle.dump((tangent_colors,D, d, M, P), f)


def load_data(save_dir, suffix):

    # Loading the files with full paths
    with open(os.path.join(save_dir, f'errors_{suffix}.pkl'), 'rb') as f:
        mean_MT_SE, mean_data_SE, all_MT_SE = pickle.load(f)

    errors = [mean_MT_SE, mean_data_SE, all_MT_SE]

    with open(os.path.join(save_dir, f'local_params_{suffix}.pkl'), 'rb') as f:
        Q, T, S_collection, P, Xi = pickle.load(f)
    local_params = [Q, T, S_collection, P, Xi]

    with open(os.path.join(save_dir, f'nbrs_info_{suffix}.pkl'), 'rb') as f:
        N1, W1, N0, W0 = pickle.load(f)
    nbrs_info = N1, W1, N0, W0

    with open(os.path.join(save_dir, f'misc_{suffix}.pkl'), 'rb') as f:
        tangent_colors, D, d, M, P = pickle.load(f)
    misc = [tangent_colors, D, d, M, P]

    with open(os.path.join(save_dir, f'time_array_{suffix}.pkl'), 'rb') as f:
        time_array = pickle.load(f)
        
    network_params = [local_params, nbrs_info, misc]

    return errors, network_params, time_array