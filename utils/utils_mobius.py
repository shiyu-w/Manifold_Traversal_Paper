import numpy as np
import plotly.graph_objects as go

def mobius_strip(s, t, R=1):
    '''
        R ----- midcircle radius
    '''
    x = (R + (s/2) * np.cos(t/2)) * np.cos(t)
    y = (R + (s/2) * np.cos(t/2)) * np.sin(t)
    z = (s/2) * np.sin(t/2)
    return np.array([x, y, z])

def distance_squared(P, s, t):
    P_bar = mobius_strip(s, t)
    return 0.5*np.sum((P_bar - P)**2)

def distance_function_for_points(x, x_bar):
    '''
        This function computes the objective value when the objective is the distance function.
        To find a projection of a point onto a manifold, we solve the optimization problem
                    \arg \min_{\bar x in \mathcal{M}} \varphi_x(\bar x),
        where \varphi_x(\bar x) = 0.5* \|\bar x - x\|_2^2.

        Inputs:
            x -------------- a 3-dim numpy array, the given point we are tyring to find the projection of.
            x_bar ---------- the projection of x onto the manifold


        Outputs:
            y -------------- a scalar representing the distance between x and its projection onto the manifold

    '''

    # use rounding so small differences don't throw an error
    # y = round(0.5*(np.linalg.norm(x-x_bar)**2), 4)
    y = 0.5*((np.linalg.norm(x-x_bar))**2)

    return y


def gradient(s, t, P):
    h = 1e-8  # Small value for numerical gradient

    # Find the gradient using finite differences -- using central difference method
    d_phi_d_t = (distance_squared(P, s, t + h) - distance_squared(P, s, t - h)) / (2 * h)
    d_phi_d_s = (distance_squared(P, s + h, t) - distance_squared(P, s - h, t)) / (2 * h)
    return np.array([d_phi_d_s, d_phi_d_t])



def gradient_by_hand(s, theta, P, R = 1):
    P_bar = mobius_strip(s, theta, R)
    # P_bar = np.clip(M_bar, a_max = np.max(M_bar), a_min=1e-6)

    d_P_bar_ds = np.array([[np.cos(theta/2)*np.cos(theta)],
                           [np.cos(theta/2)*np.sin(theta)],
                           [np.sin(theta/2)]]).reshape(-1,1)

    d_P_bar_dt = np.array([[-R*np.sin(theta) - 0.5*s*np.sin(theta/2)*np.cos(theta)-s*np.cos(theta/2)*np.sin(theta)],
                        [R*np.cos(theta) - 0.5*s*np.sin(theta/2)*np.sin(theta) + s*np.cos(theta/2)*np.cos(theta)],
                        [0.5*s*np.cos(theta/2)]]).reshape(-1,1)

    d_phi_d_s = np.dot((P_bar - P),d_P_bar_ds).reshape(-1,1)
    d_phi_d_t = np.dot(P_bar - P, d_P_bar_dt).reshape(-1,1)

    final_grad = np.array([d_phi_d_s, d_phi_d_t]).squeeze()

    return final_grad



def project_point(P, learning_rate=0.1, beta_rate = 0.1, max_iterations=1000, tolerance=1e-6, R=1,
                  finite_diff_grad = False, use_momentum = True,
                  t = np.pi,
                  s = 0):
    '''
        Performs gradient descent.

        Inputs:
            P ----------------- the point to be projected onto the mobius strip
            learning rate ----- learning rate of GD
            beta_rate --------- if using GD with momentum, beta_rate is beta*(x_{k} - x_{k-1})
            max_iterations ---- max number of iteraitons of GD
            tolerance --------- once the norm of the gradient reaches tolerance, stop the alg
            finite_diff_grad -- True of want to use gradient calculated by finite differences
            use_momentum ------ True if want to use GD with momentum istead of regular GD
            t ----------------- initial value of parameter t
            s ----------------- initial value of parameter s
    '''



    # make grad_list to be the same length as path_list
    grad_list = [np.array([1,1]), np.array([1, 1])]
    # Initial guess: middle of the parameter ranges
    # t, s = np.pi, 0

    # If we want to plot the path gradient takes, we keep track of all (t,s) pairs
    # during gradient descent in a list called path
    obj_list = []

    # create a list that will collect all points. Used to plot the path of GD later
    path = [[s, t], [s,t]]
        
    
    for i in range(max_iterations):
        if finite_diff_grad:
            grad = gradient(s, t, P)
        else:
            grad = gradient_by_hand(s, t, P, R)
        
        if use_momentum:
            s_new = s - learning_rate * grad[0] + beta_rate * (s - path[-2][0])
            t_new = t - learning_rate * grad[1] + beta_rate * (t - path[-2][1])


        else:
            # Update s and t
            s_new = s - learning_rate * grad[0]
            t_new = t - learning_rate * grad[1]
        
        # Ensure t stays within [0, 2pi] and s within [-1, 1] (or [-w,w])
        #t_new = t_new % (2 * np.pi)
        s_new = np.clip(s_new, -1, 1)

        # Update parameters
        s, t = s_new, t_new

        # compute the objective and save it in a list
        obj = distance_squared(P, s, t)
        obj_list.append(obj)

        # Append the next point of GD journey onto the path list
        path.append([s_new, t_new])

        grad_list.append(grad)

        # Check for convergence
        # if np.linalg.norm([t_new - t, s_new - s]) < tolerance:
        #     break

        # Check for convergence. If norm of the gradient is small
        if np.linalg.norm(grad) < tolerance:
            # print('MET OBJ WITH S and THETA = ', s_new, t_new)
            # print('MET OBJ VALUE AT ITERATION {}'.format(i))
            # print('EXITING.....')
            break
    
    # Clip to avoid really small zero values
    P_bar = mobius_strip(s, t, R)
    # print('FINAL ITERAITON = ', i)

    return P_bar, path, s, t, grad_list, obj_list
    

def generate_tangent_space_to_mobius(P_bar, s, theta, R = 1):
    '''
        Given the point P_bar on the mobius strip, this function returns
        the tangent space to the mobius strip at that point. It does so by
        \mathcal{T}_{P_bar} \mathcal {M} = span{\frac{d P_bar}{d s}, \frac{d P_bar}{d \theta}}.

        Inputs:
            P_bar ---------- ndarray, a point in 3D on the Mobius strip
            s -------------- parameter of the point where we are drawing the tangent space to
            theta ---------- parameter of the point where we are drawing the tangent space to
            R -------------- parameter, normally set to 1
    '''

    # Define the basis vectors
    u_1 = np.array([np.cos(theta / 2) * np.cos(theta),
                    np.cos(theta/2) * np.sin(theta),
                    np.sin(theta/2)])
    
    u_2 = np.array([-R * np.sin(theta) - 0.5 * s * np.sin(theta/2)*np.cos(theta) - s*np.cos(theta/2)*np.sin(theta),
                    R * np.cos(theta) - 0.5 * s* np.sin(theta/2)*np.sin(theta) + s*np.cos(theta/2)*np.cos(theta),
                    0.5 * s * np.cos(theta/2)])
    
    # Noralize basis vectors
    u1 = u_1 / np.linalg.norm(u_1)
    u2 = u_2 / np.linalg.norm(u_2)

    # Generate a grid of points
    grid_size = 10
    s_vals = np.linspace(-1, 1, grid_size)
    t_vals = np.linspace(-1, 1, grid_size)
    s_grid, t_grid = np.meshgrid(s_vals, t_vals)

    # Transform the grid points to lie on the plane
    x = s_grid * u1[0] + t_grid * u2[0] + P_bar[0]
    y = s_grid * u1[1] + t_grid * u2[1] + P_bar[1]
    z = s_grid * u1[2] + t_grid * u2[2] + P_bar[2]

    return x, y, z



def monte_carlo_initialization(P, num_samples, num_local_mins, theta_offset = 4e-7, s_offset = 1e-6, learning_rate = 0.01,
                               max_iterations = 1000, tolerance = 1e-6, R = 1,
                               beta_rate = 0.1, finite_diff_grad = False, use_momentum = False):
    '''
        This function uses monte carlo initialization for gradient descent to avoid being
        stuck in local minima. The mobius strip has basins of attraction due to the twists.
        We initialize uniformly at random in the intercal of parameters, but we also initialize
        uniformly at random within the basins of attractions. We do so by identifying the 
        intersection of the normal space with the manifold, and initializing in the small 
        areas near these intersections. 

        Inputs:
            P ------------------- the given point to be projected onto the manifold
            num_samples --------- number of initializations in the entire interval of parameters
            num_local_mins ------ number of initializations within the basins of attraction that
                                    result in a local min. Each basin gets num_local_mins//2 
                                    ininitalizations.

            theta_offset -------- the ball around intersection of the normal space with the manifold
                                    is of size theta_offset for parameter theta. Note that this ball
                                    is very small. The basin of attraction that leads to a local min
                                    is small.
            s_offset ------------ same as theta_offset but for the s parameter
            learning_rate ------- stepsize of the GD
            max_iterations ------ max number of iterations
            tolerance ----------- once the norm of the gradient is less than tolerance, we stop the alg
            R ------------------- mobius strip parameter

        Outputs:
            final_list_of_params - a list of parameters of the form [[s1, t1], ..., [sn, tn]]. All these
                                    pairs are the result of GD initalized in basins ofm local mins, as 
                                    well as over the entire parameter space. 
    '''
        

    # find angles that will result in local mins
    theta = np.arctan2(P[1], P[0])
    theta_2 = theta + np.pi

    list_of_local_min_initializations = []
    final_list_of_params = []

    path_lm1 = []
    # initialize in the first local min basin of attraction several times
    for i in range(num_local_mins//2):
        # inrialize parameters for local min points
        t_lm_init  = np.random.uniform(theta - theta_offset, theta + theta_offset)
        s_lm_init  = np.random.uniform(-s_offset, s_offset)

        # save initializations in a list
        list_of_local_min_initializations.append([s_lm_init, t_lm_init])

        # print('s_lm_init, t_lm_init = ', s_lm_init, t_lm_init)

        # s_lm_final, t_lm_final are the final parameters obtained by GD when initialized in the local min basin
        P_bar, path_lm1_, s_lm_final, t_lm_final, _, _ = project_point(P, learning_rate = learning_rate, beta_rate = beta_rate,
                                                               max_iterations=max_iterations, tolerance=tolerance, R = R,
                                                               finite_diff_grad = finite_diff_grad, use_momentum = use_momentum,
                                                               t = t_lm_init,
                                                               s = s_lm_init)

        final_list_of_params.append([s_lm_final, t_lm_final])
        path_lm1.append(path_lm1_)
        
    # print('path_lm1 = ', path_lm1)

    # print('-----------------------------------------------------------------------------')

    # initialize in the second local min basin of attraction several times with theta + pi
    # s_lm_final, t_lm_final are the final parameters obtained by GD when initialized in the local min basin
    path_lm2 = []
    for i in range(num_local_mins//2):
        # inrialize parameters for local min points
        t_lm_init  = np.random.uniform(theta_2 - theta_offset, theta_2 + theta_offset)
        s_lm_init  = np.random.uniform(-s_offset, s_offset)

        # save initializations in a list
        list_of_local_min_initializations.append([s_lm_init, t_lm_init])

        # print('s_lm_init, t_lm_init = ', s_lm_init, t_lm_init)

        P_bar, path_lm2_, s_lm_final, t_lm_final, _, _ = project_point(P, learning_rate = learning_rate, beta_rate = beta_rate,
                                                               max_iterations=max_iterations, tolerance=tolerance, R = R,
                                                               finite_diff_grad = finite_diff_grad, use_momentum = use_momentum,
                                                               t = t_lm_init,
                                                               s = s_lm_init)

        final_list_of_params.append([s_lm_final, t_lm_final])
        path_lm2.append(path_lm2_)

    # print('-----------------------------------------------------------------------------')
    # print('-----------------------------------------------------------------------------')
    # print('-----------------------------------------------------------------------------')

    list_of_regular_initializations = []
    path = []
    for i in range(num_samples):

        t_init = np.random.uniform(0, 2 * np.pi)
        s_init = np.random.uniform(-1, 1)

        # save initializations in a list
        list_of_regular_initializations.append([s_init, t_init])
        # print('s_init, t_init = ', s_init, t_init)

        P_bar, path_, s_final, t_final, _, _ = project_point(P, learning_rate = learning_rate, beta_rate = beta_rate,
                                                         max_iterations = max_iterations, tolerance = tolerance, R = R,
                                                         finite_diff_grad = finite_diff_grad, use_momentum = use_momentum,
                                                         t = t_init,
                                                         s = s_init)

        final_list_of_params.append([s_final, t_final])
        path.append(path_)

    return final_list_of_params, list_of_local_min_initializations, list_of_regular_initializations, theta, theta_2, [path_lm1, path_lm2, path]


def get_unique_sublists(nested_list, dec = 3):

    '''
        This function takes a list of the form [[s1,t1], [s2,t2], ..., [sn,tn]] and gets rid of duplicates up
        to a precision of dec = 3 decimals. Returns a list of pairs which unique values.
    '''

    # first, round the numbers up to the 4th decimal, otherwise we get too many distinct points
    rounded_nested_list = np.round(nested_list, dec)

    # Convert each sublist to a tuple
    tuple_list = [tuple(sublist) for sublist in rounded_nested_list]

    # Use a set to keep only unique tuples
    unique_tuples = set(tuple_list)
    
    # Convert the unique tuples back to lists
    unique_sublists = [list(t) for t in unique_tuples]

    return unique_sublists



def pick_pt_with_smallest_dist(P, num_samples = 200, num_local_mins = 50, theta_offset = 4e-7, s_offset = 1e-6, dec = 3,
                               learning_rate = 0.01, max_iterations = 1000, tolerance = 1e-6, R = 1):
    P_x, P_y, P_z = P

    # find the final list of parameter pairs. GD converges to this list
    final_list_of_params, list_of_local_min_initializations, list_of_regular_initializations,\
        theta, theta_2, [path_lm1, path_lm2, path] = monte_carlo_initialization(P,
                                                                num_samples = num_samples, num_local_mins = num_local_mins,
                                                                theta_offset = theta_offset, s_offset = s_offset,
                                                                learning_rate = learning_rate, max_iterations = max_iterations,
                                                                tolerance = tolerance, R = R)
    
    # first we have a list of the form [[s1,t1], ..., [sn,tn]]. Each pair is the final parameters given by the GD
    # initialized at local min basins as well as over the entire parameter space. The list has many values that 
    # are very close to each other. We choose to pick out the unique ones, up to a certain decimal of precision.

    # boil it down to a list of points with unique (s,t) parameters
    unique_sublists = get_unique_sublists(final_list_of_params, dec = dec)
    # print(unique_sublists)

    list_of_distances = []

    for param in unique_sublists:
        # compute the distance between the given point and the current point in the list
        cur_dist = distance_squared(P = P, s = param[0], t = param[1])
        list_of_distances.append(cur_dist)

    # find the index of the smallest distance
    idx = np.argmin(list_of_distances)

    # save the point that gave the smallest distance. This is the final projection
    optimal_point_params = unique_sublists[idx]

    return optimal_point_params, unique_sublists, final_list_of_params, list_of_local_min_initializations, list_of_regular_initializations, theta, theta_2, [path_lm1, path_lm2, path]


