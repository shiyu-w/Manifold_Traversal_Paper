import numpy as np
from utils.utils_demo_growth import main_fn_no_vis
import time


def train_network(N_start, N_end, errors, cur_network_params, X, X_natural, Rd_params, other_params):
    """
    Trains on some samples
    Input: 
        - N_start = starting index of training samples for this batch
        - N_end = ending index of training samples for this batch
        - cur_network_params = [local_params, nbrs_info, misc]
            - local_params = [Q, T, s_collection, P, Xi]
            - nbrs_info = [N1, W1, N0, W0]
            - misc = [tangent_colors, D, d, M, P]

    Output:
        - total_time = the time it took to train with this batch of data
        - errors
        - new_network_params = [new_local_params, new_nbrs_info, new_misc] (after output)
            - new_local_params = [Q, T, P, Xi]
            - new_nbrs_info = [N1, W1, N0, W0]
            - new_misc = [tangent_colors, D, d, M, P]

    Assumes the following params are floating around in the Jupyter Notebook:
    trunc_waves, manifold, N=N, manifold_type, output_path, R_denoising, R_1st_order_nbhd,d,D,sigma, X, X_natural

    """
    # load params
    [mean_MT_SE, mean_data_SE, all_MT_SE] = errors
    [local_params, nbrs_info, misc]       = cur_network_params
    [Q, T, S_collection, P, Xi]           = local_params
    [N1, W1, N0, W0]                      = nbrs_info
    [tangent_colors, D, d, M, P]          = misc

    (sigma, R_1st_order_nbrs) = other_params
    (R_is_const, R_denoising, d_parallel, prod_coeff, exp_coeff) = Rd_params

    start_time = time.time()

    errors, new_local_params, new_nbrs_info, new_misc = main_fn_no_vis(X, X_natural, sigma=sigma,
            N = N_end,
            d_parallel = d_parallel,
            R_denoising = R_denoising,
            R_1st_order_nbhd = R_1st_order_nbrs,
            d = d, D = D,
            N_cur = N_start,
            all_MT_SE = all_MT_SE,
            mean_MT_SE     = mean_MT_SE, 
            mean_data_SE   = mean_data_SE,
            frame_num = 0,
            M = M,
            Q = Q,
            T = T,
            S_collection = S_collection,
            P = P,
            N1 = N1,
            W1 = W1,
            Xi = Xi,
            N0 = N0,
            W0 = W0,
            tangent_colors = tangent_colors,
            R_is_const = R_is_const,
            prod_coeff = prod_coeff,
            exp_coeff = exp_coeff)


    end_time = time.time()
    total_time = end_time - start_time

    print('TOTAL TIME = ', total_time)
    new_network_params = [new_local_params, new_nbrs_info, new_misc]
    return [total_time, errors, new_network_params]






def train_network_wrapper(R_is_const, R_denoising, R_1st_order_nbrs, d_parallel, prod_coeff, exp_coeff, name, 
                          D, d, sigma, 
                          N_train, X_train, X_natural_train, batch_size=4000):
    # initialize an MTN object 
    M = 0 # number of landmarks

    "local approximation info"
    Q = [] # list of landmarks w
    T = [] # list of basis matrices
    S_collection = [] # a list of sigma matrices obtained after each TISVD

    P = [] # number of points in each approximation 

    "first order graph info"
    N1 = [] # list of lists of first order neighbors
    W1 = [] # list of lists of weights of first order neighbors 
    Xi = [] # list of lists of edge embeddings 

    "zero order graph info" 
    N0 = [] # list of lists of zero order neighbors
    W0 = [] # list of lists of weights of zero order neighbors 
    tangent_colors = []

    local_params = [Q, T, S_collection, P, Xi]
    nbrs_info = [N1, W1, N0, W0]
    misc = [tangent_colors, D, d, M, P]
    network_params = [local_params, nbrs_info, misc]
    N_cur = 0

    all_MT_SE      = []
    mean_MT_SE     = [] 
    mean_data_SE   = []

    errors = [mean_MT_SE, mean_data_SE, all_MT_SE]
    other_params = (sigma, R_1st_order_nbrs)
    Rd_params = (R_is_const, R_denoising, d_parallel, prod_coeff, exp_coeff)
    
    # train online method in batches:
    # NOTE: the batches' only purpose is to notify the user regarding the progress of learning.
    # there is no algorithmic implication to training in batches
    # the data is still fed individually in an online fashion, except that the time for processing "batch_size" consecutive samples is tracked and printed
    # batch_size = 4000 #set batch_size as desired
    num_batches = N_train // batch_size
    time_array = []
    errors_array = []
    for i in range(num_batches):
        start_index = i * batch_size
        end_index = start_index + batch_size
        cur_time, errors, network_params = train_network(start_index, end_index, errors, network_params, X = X_train, X_natural = X_natural_train,
                                                            other_params=other_params, Rd_params=Rd_params)
        time_array.append(cur_time)
        errors_array.append(errors)
        print(f"{end_index} samples processed...")
    
    print(f"DONE... TOTAL TIME = {np.sum(np.array(time_array))}")
    return network_params, errors