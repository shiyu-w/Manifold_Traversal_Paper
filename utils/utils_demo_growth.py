# updated Feb 6, 2025

# math
import numpy as np
import math

# plotting 
import matplotlib.pyplot as plt 
import matplotlib.colors as mcolors
all_colors = list(mcolors.CSS4_COLORS)

from utils.utils_TISVD import TISVD_gw


import random

from sklearn.decomposition import TruncatedSVD
from scipy.linalg import svd



N0 = []
N1 = []


def R_denoising_sq(sigma, D, d_parallel, P_i, R_is_const = True, prod_coeff = 1.2, exp_coeff = 1/2, R_denoising = 0.65):
    '''
        This function returns either a fixed or decreasing value of R_denoising^2.
    '''
    if R_is_const:
        return R_denoising**2
    else:
        return prod_coeff * (sigma**2 *D + (sigma**2*D / (P_i**exp_coeff)) + d_parallel**2)
    


def MT_perform_traversal(x, Q, T, N1, W1, Xi, N0, W0, calc_mults = True):

    '''
        Inputs:
            x ------------ the given point to be denoised
            Q ------------ landmarks
            T ------------ a matrix of tangent space basis vectors
            N1 ----------- a list of lists indicating first-order edges?
            W1 ----------- weight matrix for 1st-order edges
            Xi ----------- a matrix of the psi edge embeddings in the tangent space of vertex i ?
            N0 ----------- number of zero-order edges?
            W0 ----------- weight matrix for zero-order edges ?


        Outputs:
            i ------------ the final vertex
            phi ---------- the final objective value
            trajectory --- the list of vertices visited
            edge_orders -- list of edge orders used (0 or 1) for each step in the trajectory   
            mults -------- the number of multiplications performed (0 if calc_mults = False)
    '''
    mults = 0
    if calc_mults == True:
        D = len(x)
        d = len(T[0][0])

    i = 0  # starting vertex

    converged = False
    iter = 0

    trajectory = [i]
    edge_orders = []

    phi = np.sum( (Q[i] - x) ** 2 ) #loss function value at the initialization point
    if calc_mults:
        mults += D 

    while not converged: 
        # calculate and # print current objective 
        # phi = np.sum( (Q[i] - x) ** 2 )

        # if calc_mults:
        #     mults += D 

        # compute Riemannian gradient in coordinates
        # this is the gradient of .5 || q - x ||_2^2 with respect to q
        grad_phi = T[i].transpose() @ ( Q[i] - x )

        if calc_mults:
            mults += D*d

        # 1st order outdegree of vertex i
        deg_1_i = len( N1[i] )

        # find the most correlated edge embedding -- this is the speculated next vertex
        next_i = 0
        best_corr = math.inf

        # check the correlation for each 1st order edge of vertex i with the gradient
        for j in range( 0, deg_1_i ): 
            corr = np.dot( Xi[i][j], grad_phi )

            if (corr < best_corr):
                best_corr = corr
                next_i = N1[i][j]
        
        
        if calc_mults:
            mults += d * deg_1_i

        # compute objective value at speculated next vertex
        next_phi = np.sum( (Q[next_i] - x) ** 2 ) 

        if calc_mults:
            mults += D

        if (next_phi >= phi):
            # If the first-order step doesn't improve the objective
            # Try a zero-order step by checking all zero-order neighbors
            # Move to the neighbor with the best (lowest) objective value
            # set order = 0

            # first order step failed, try zero order step 
            best_i = 0 
            best_phi = math.inf 

            # zero-order out-degree of this vertex
            # deg_0_i = len( W0[i] )
            deg_0_i = len(N0[i])

            # compute the objective at each of the neighbors, record the best objective 
            if calc_mults:
                mults += D * deg_0_i

            for j in range( 0, deg_0_i ):
                
                cur_nbr_phi = np.sum( ( Q[ N0[i][j] ] - x ) ** 2 ) 

                if (cur_nbr_phi < best_phi):
                    best_phi = cur_nbr_phi
                    best_i = N0[i][j]

            order = 0
            next_i = best_i
            next_phi = best_phi
        else:
            # If moving to the selected neighbor improves the objective
            # Move to that neighbor (first-order step)
            # and set order = 1
            order = 1

        if (next_i == i):
            # If no improvement is found (i.e., next_i == i)
            # declare convergence and exit the loop.
            converged = True 
        else:
            # Otherwise, update i to the new vertex, append to the trajectory, and continue the loop.
            i = next_i 
            phi = next_phi 

            trajectory.append(i) 
            edge_orders.append(order) 

            iter += 1 
            
    return i, phi, trajectory, edge_orders, mults



def MT_denoise_local(x,Q,T,i):

    # denoise by projection -- this finds the closest point in Q[i] + span(T[i]) to x:
    x_hat = Q[i] + T[i] @ ( T[i].transpose() @ ( x - Q[i] ) )

    return x_hat 




def MT_initialize_new_landmark(x,Q,T,P,N1,W1,Xi,N0,W0, D, d, S_collection, sigma, d_parallel, distances_sq, R_nbrs): 
    '''
        This function attempts to do an efficient linear scan as opposed to a for
        loop done in MT_initialize_new_landmark_original().
    
    '''

    M = len(Q)
    if (M == 0):
        
        "initialize landmark and tangent space" 
        Q.append( x.copy() )                          # initialize the new landmark as the current sample 

        # SCIPY
        random_matrix = np.random.randn(D, d)
        U, s, Vt = svd(random_matrix, full_matrices=False)  # Economy SVD
        U_new = U[:, :d]  # Take first d columns
        s_new = s[:d]     # Take first d singular values
        S_new_diag = np.diag(s_new)  # If you need the diagonal matrix

        T.append( U_new )                                 # make this our initial tangent space estimate
        S_collection.append(S_new_diag)                        # save our initial matrix of singular values 
    
        P.append( 1 )                                 # number of points contributing to this model 
    
        " initialize first order graph info " 
        N1.append( [ 0   ] )                          # initially, only first order neighbor is the vertex itself  
        W1.append( [ 1.0 ] )                          # give the self-edge unit weight
        Xi.append( [ np.zeros((d,)) ] )               # initially, only a zero edge embedding for the self-edge
        # print('LITTLE d = ', d)
    
        " initialize zero order graph info " 
        N0.append( [ 0   ] )                          # initially, only zero order neighbor is the vertex itself 
        W0.append( [ 1.0 ] )                          # give the self-edge unit weight 

    else: # if we have seen more than one point

        "initialize landmark and tangent space" 
        Q.append( x.copy() )                          # initialize the new landmark as the current sample 
        
        P.append( 1 )                                 # this landmark represents 1 data point so far

        " initialize zero order graph info " 
        N0.append( [ M   ] )                          # initially, only zero order neighbor is the vertex itself 
        W0.append( [ 1.0 ] )                          # give the self-edge unit weight 
        
        "initialize first order graph info " 
        N1.append( [] )
        W1.append( [] )
        Xi.append( [] ) 
        
        # Find 1st order neighbors within radius R_1st_order_nbhd



        temp1 = np.argwhere(distances_sq <= R_nbrs**2)
        if len(temp1)==0:
            temp3 = temp1
        else:
            temp2=np.stack(temp1,axis=1)
            temp3 = temp2[0]
            
        idces = temp3

        for idx in idces:
            # this is not a self-edge ... need to also make M a neighbor of l
            N1[idx].append( M   )
            W1[idx].append( 1.0 )
            N1[M].append(idx)
            W1[M].append(1.0)

        
        N1[M].append(M)
        W1[M].append(1.0)




        # check if the new point has more than one neighbor
        # if it does, update its tangent space
        if ( len(N1[M]) > 1 ): 


            "Form H with difference vectors AND with estimated tangent spaces"
            # # iterate over all 1st-order neighbors of point M
                
                # This is what H looks like
                # H = [  Direction1 | TangentSpace1 | Direction2 | TangentSpace2 | ... ]
                #        <1 column>   <d columns>     <1 column>   <d columns>

            "Form H with difference vectors only"
            H = np.zeros( ( D, (d+1) * (len(N1[M])-1) ) )
            # iterate over all 1st-order neighbors of point M
            for l in range(0,len(N1[M])-1):
                H[ :, l ] = ( Q[N1[M][l]] - Q[M] ) / np.linalg.norm( Q[N1[M][l]] - Q[M] )

            # SCIPY
            U, s, Vt = svd(H, full_matrices=False)  # Economy SVD
            U_new = U[:, :d]  # Take first d columns
            s_new = s[:d]     # Take first d singular values
            S_new_diag = np.diag(s_new)  # If you need the diagonal matrix

            # update the tangent space
            T.append( U_new ) # make this our initial tangent space estimate
            S_collection.append(S_new_diag)    


        else: # if the point has no neighbors, just itself
            # no neighbors, just use a random subspace ... 
            # SCIPY
            random_matrix = np.random.randn(D, d)
            U, s, _ = svd(random_matrix, full_matrices=False)  # Economy SVD
            U_new = U[:, :d]  # Take first d columns
            s_new = s[:d]     # Take first d singular values
            S_new_diag = np.diag(s_new)  # If you need the diagonal matrix

            T.append( U_new )                                 # make this our initial tangent space estimate
            S_collection.append(S_new_diag)                        # save our initial matrix of singular values 
        


        # JW: can condense these loops to just use the neighbor information, rather than recompute locality 
        for l in range(0,len(N1[M])):
            Xi[M].append( T[M].transpose() @ ( Q[N1[M][l]] - Q[M] ) )
            Xi[N1[M][l]].append( T[l].transpose() @ ( Q[M] - Q[N1[M][l]] ) )





def MT_initialize_new_landmark_original(x,Q,T,P,N1,W1,Xi,N0,W0,tangent_colors,R_1st_order_nbhd, D, d, S_collection): 

    M = len(Q)
    if (M == 0): 
        
        "initialize landmark and tangent space" 
        Q.append( x.copy() )                          # initialize the new landmark as the current sample
        # SCIPY
        random_matrix = np.random.randn(D, d)
        U, s, Vt = svd(random_matrix, full_matrices=False)  # Economy SVD
        U_new = U[:, :d]  # Take first d columns
        s_new = s[:d]     # Take first d singular values
        S_new_diag = np.diag(s_new)  # If you need the diagonal matrix

        T.append( U_new )                                 # make this our initial tangent space estimate
        S_collection.append(S_new_diag)                   # save our initial matrix of singular values 
    
        P.append( 1 )                                 # number of points contributing to this model 
    
        " initialize first order graph info " 
        N1.append( [ 0   ] )                          # initially, only first order neighbor is the vertex itself  
        W1.append( [ 1.0 ] )                          # give the self-edge unit weight
        Xi.append( [ np.zeros((d,)) ] )               # initially, only a zero edge embedding for the self-edge
    
        " initialize zero order graph info " 
        N0.append( [ 0   ] )                          # initially, only zero order neighbor is the vertex itself 
        W0.append( [ 1.0 ] )                          # give the self-edge unit weight 

        # " give it a random color for visualization " 
        tangent_colors.append( all_colors[ np.random.randint(0,len(all_colors)) ] ) 

    else: # if we have seen more than one point

        "initialize landmark and tangent space" 
        Q.append( x.copy() )                          # initialize the new landmark as the current sample 
        P.append( 1 )                                 # this landmark represents 1 data point so far

        " initialize zero order graph info " 
        N0.append( [ M   ] )                          # initially, only zero order neighbor is the vertex itself 
        W0.append( [ 1.0 ] )                          # give the self-edge unit weight 
        
        "initialize first order graph info " 
        N1.append( [] )
        W1.append( [] )
        Xi.append( [] ) 

        " give it a random color for visualization " 
        tangent_colors.append( all_colors[ np.random.randint(0,len(all_colors)) ] )


        
        # Find 1st order neighbors within radius R_1st_order_nbhd
        for l in range(0,M+1):
            if ( np.sum( ( Q[M] - Q[l] ) ** 2 ) <= R_1st_order_nbhd * R_1st_order_nbhd ): 
                # l is a neighbor of M
                N1[M].append( l   )
                W1[M].append( 1.0 )

                if (l < M):  

                    # this is not a self-edge ... need to also make M a neighbor of l
                    N1[l].append( M   )
                    W1[l].append( 1.0 )



        # check if the new point has more than one neighbor
        # if it does, update its tangent space
        if ( len(N1[M]) > 1 ): 
            # len(N1[M])-1 is number of neighbors (excluding self)
            "Form H with difference vectors only"
            H = np.zeros( ( D, (d+1) * (len(N1[M])-1) ) )
            # iterate over all 1st-order neighbors of point M
            for l in range(0,len(N1[M])-1):
                H[ :, l ] = ( Q[N1[M][l]] - Q[M] ) / np.linalg.norm( Q[N1[M][l]] - Q[M] )
                # This is what H looks like
                # H = [  Direction1 | Direction2 | ... ]

            # SCIPY
            U, s, Vt = svd(H, full_matrices=False)  # Economy SVD
            U_new = U[:, :d]  # Take first d columns
            s_new = s[:d]     # Take first d singular values
            S_new_diag = np.diag(s_new)  # If you need the diagonal matrix

            # update the tangent space
            T.append( U_new ) # make this our initial tangent space estimate
            S_collection.append(S_new_diag)    


        else: # if the point has no neighbors, just itself
            # no neighbors, just use a random subspace ...
            # SCIPY
            random_matrix = np.random.randn(D, d)
            U, s, _ = svd(random_matrix, full_matrices=False)  # Economy SVD
            U_new = U[:, :d]  # Take first d columns
            s_new = s[:d]     # Take first d singular values
            S_new_diag = np.diag(s_new)  # If you need the diagonal matrix

            T.append( U_new )                                 # make this our initial tangent space estimate
            S_collection.append(S_new_diag)                        # save our initial matrix of singular values 
        
        for l in range(0,len(N1[M])):
            # edge embeddings
            Xi[M].append( T[M].transpose() @ ( Q[N1[M][l]] - Q[M] ) )
            Xi[N1[M][l]].append( T[l].transpose() @ ( Q[M] - Q[N1[M][l]] ) )





def MT_update_local_SGD_TISVD(x, Q, T, P, Xi, i, d, N1, S_collection): 
    '''
        Update the local parameters using truncated incremental SVD per sample (not minibatch).

        Inputs:
            x -------------------- current point
            Q -------------------- collecgtion of landmarks
            T -------------------- a collection of tangent space basis vectors
            P -------------------- counter for the number of updates at each point (keeps count of # of pts assigned per landmark)
            Xi ------------------- collection of edge embeddings
            i -------------------- index of the current point
            d -------------------- intrinsic dimension
            N1 ------------------- a list of lists, the ith list contains indices of first-order neighbors of landmark q_i
            S_collection --------- a list of singular value diagonal matrices
        Outputs:
            None
    
    '''

    P[i] += 1

    Q[i] = (( P[i] - 1.0 ) / P[i] ) * Q[i] + ( 1.0 / P[i] ) * x

    U_old = T[i-1]
    S_old = S_collection[i-1]
    U_new, S_new_diag = TISVD_gw(x, U_old, S_old, i, d)

    T[i] = U_new.copy()

    S_collection[i] = S_new_diag.copy()
    
    # recompute edge embeddings 
    for j in range(0,len(N1[i])):
        i_pr = N1[i][j]
        Xi[i][j] = T[i].transpose() @ ( Q[i_pr] - Q[i] )
        for l in range(0,len(N1[i_pr])):
            if ( N1[i_pr][l] == i ):
                Xi[i_pr][l] = T[i_pr].transpose() @ ( Q[i] - Q[i_pr] )


def main_fn_no_vis(X, X_natural,
            sigma,
            N,
            d_parallel = 0.01414,
            R_denoising = 0.4, R_1st_order_nbhd = 0.8,
            d = 2, D = 3,
            N_cur = 0,
            all_MT_SE = [],
            mean_MT_SE     = [], 
            mean_data_SE   = [],
            frame_num = 0,
            M = 0, Q = [], T = [], S_collection = [], P = [],
            N1 = [], W1 = [], Xi = [], N0 = [], W0 = [], tangent_colors = [],
            R_is_const = True,
            prod_coeff = 1.2,
            exp_coeff = 1/2,
            calc_mults = True):
    '''
        This function runs the online algorithm.
        Inputs:
            X ------------ noisy dataset of size D x N
            X_natural ---- clean samples of size D x N
    '''
    while (N_cur < N):
        # grab the current sample 
        x = X[:,N_cur]

        if (M == 0):

            # current traversal network is empty -- let's use this sample to initialize a landmark
            MT_initialize_new_landmark_original(x,Q,T,P,N1,W1,Xi,N0,W0,tangent_colors,R_1st_order_nbhd, D, d, S_collection)
            
            M += 1
            x_hat = x

        else:

            " Current Network Structure "
            # i is a landmark index
            # phi is the squared distance ||x - Q[i]||_2^2
            # i, phi, trajectory, order, _ = MT_perform_traversal(x,Q,T,N1,W1,Xi,N0,W0)
            i, phi, trajectory, edge_orders, mults = MT_perform_traversal(x,Q,T,N1,W1,Xi,N0,W0, calc_mults)

            " Evaluate the result of traversal and update network parameters " 
            R_d_sq = R_denoising_sq(sigma,D,d_parallel,P[i],R_is_const, prod_coeff,exp_coeff,R_denoising)

            if ( phi <= R_d_sq):
                # this sample is an inlier ... perform denoising and update model parameters 
                # denoise by projection onto local model at the i-th landmark 
                x_hat = MT_denoise_local(x,Q,T,i) 
                            
                # update local model around the i-th landmark
                MT_update_local_SGD_TISVD(x,Q,T,P,Xi,i,d, N1, S_collection)
                        
            else:
                # perform exhaustive search 
                i_best = 0 
                best_phi = math.inf 

                # compute the objective at each of the neighbors, record the best objective 
                for j in range( 0, M ):
                    cur_phi = np.sum( ( Q[ j ] - x ) ** 2 ) 
                    if (cur_phi < best_phi): 
                        best_phi = cur_phi
                        i_best = j 

                phi = best_phi

                R_d_sq = R_denoising_sq(sigma,D,d_parallel,P[i_best],R_is_const, prod_coeff,exp_coeff,R_denoising)
                if ( phi <= R_d_sq): 

                    # add the neighbor, with weight 1 
                    N0[i].append(i_best)
                    W0[i].append( 1.0 )

                    # denoise by projection onto local model at the i-th landmark 
                    x_hat = MT_denoise_local(x,Q,T,i_best) 
        
                    # update local model around the i-th landmark
                    MT_update_local_SGD_TISVD(x,Q,T,P,Xi,i,d,N1,S_collection)

                else: 
                    MT_initialize_new_landmark_original(x,Q,T,P,N1,W1,Xi,N0,W0,tangent_colors,R_1st_order_nbhd, D, d, S_collection)
                    # no local model, so just make the trivial prediction
                    x_hat = x
                    M += 1 

            frame_num += 1
 

        cur_MT_SE = np.sum( ( x_hat - X_natural[:,N_cur] ) ** 2 )
        cur_data_SE = np.sum( ( x - X_natural[:,N_cur] ) ** 2 )
        
        all_MT_SE.append( cur_MT_SE )
        if (N_cur == 0):
            mean_MT_SE.append( cur_MT_SE )
            mean_data_SE.append( cur_data_SE )
        else: 
            mean_MT_SE.append( (1.0/(N_cur+1.0)) * cur_MT_SE + (N_cur/(N_cur+1.0)) * mean_MT_SE[ N_cur - 1 ] )
            mean_data_SE.append( (1.0/(N_cur+1.0)) * cur_data_SE + (N_cur/(N_cur+1.0)) * mean_data_SE[ N_cur - 1 ] )

        N_cur += 1

    return (mean_MT_SE, mean_data_SE, all_MT_SE), (Q,T,S_collection,P,Xi), (N1,W1,N0,W0), (tangent_colors,D, d, M, P)







