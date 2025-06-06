import numpy as np

# Function to generate Swiss roll data
def swiss_roll_parametrization(t, v, v_min, v_max):
    # parametric equations of the swiss roll
    x = t * np.cos(t)
    # z = v
    z = np.clip(v, v_min, v_max)
    y = t * np.sin(t)

    return np.array([x, y, z]).T



def find_angles_swiss_roll(P):

    '''
    Given a point P in 3D, find the angle theta. To be used as an input in go_around_unit_circle().
    We want theta to be positive because, in the swiss roll, the parameter t is in the range [0, N*pi],
    so it is not useful to have negative thetas.

    Returns angles in radians.
    '''
    # Unpack the given point
    P_x, P_y, P_z = P

    theta = np.arctan2(P_y , P_x)

    # keep theta positive by adding 2pi to the negative values
    if theta >= 0:
        theta_positive = theta
    else:
        theta_positive = theta + 2 * np.pi

    return theta_positive


def go_around_unit_circle(theta, p = np.pi, theta_max = 10 * np.pi, show_in_degrees = False):
    '''
        This function, given an angle theta, finds all $theta \pm k * pi \in [0, theta_max]$.
        In other words, angles representing all points of intersection of the swiss roll with
        the normal space at P_bar, i.e. the projection of P onto the roll.
        For example, if theta is pi/4, and theta_max = 4 pi, then this function will output
        pi/4, pi/4 + pi, pi/4 + 2pi, pi/4 + 3pi.

        Inputs:
            theta ------------- angle found based on a given point x whose projection onto
                                the swiss roll we are trying to find
            p ----------------- since we are working with the unit circle, this will be pi
            theta_max --------- decides size of the interval [0, theta_max], which in turn
                                decides the number / shape of the spiral of the swiss roll
            show_in_degrees --- True; for testing, we normally want to return the values in
                                degrees rather than radians so it's easier to troubleshoot

        Outputs:
            list_of_angles ---- a list of angles that lie within the interval
            remainder --------- remainder of theta modulo pi

    '''

    quotient = int(theta // p)
    remainder = theta - quotient * p

    if show_in_degrees:
        print('quotient = ', np.degrees(quotient))
        print('remainder = ', np.degrees(remainder))

    list_of_angles = []
    current_theta = remainder

    while current_theta <= theta_max:
        list_of_angles.append(current_theta)
        current_theta = list_of_angles[-1] + p

    # print('list_of_angles = ', list_of_angles)

    if show_in_degrees:
        return np.degrees(list_of_angles), np.degrees(remainder)
    else:
        return list_of_angles, remainder
    



def distance_function(x, x_bar):
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



def find_P_bar_init_swiss_roll(P = np.array([0,0,0]), v_min = 0, v_max = 10, theta_max = 10 * np.pi):
    '''
        This is a function for finding a suitable initial point on the swiss roll (i.e.
        a quick-and-dirty projection). This point is then used as initialization in the
        gradient descent algorithm.
        
        It takes a point P and returns P_bar_init as a 3-dimensional ndarray. Note that
        P_bar_init is not the projection of P. We use P_bar_init as an initialization for 
        gradient descent in project_pont() function later.

        Inputs:
            P ---------------- the point to be projected onto the manifold
            v_min, v_max ----- range of the parameter v of the swiss roll
            theta_max -------- parameter t of the swiss roll ranges from [0, theta_max]

        Outputs:
            P_bar_init ------- the "quick-and-dirty" projection of the point P onto the swiss
                                roll. This is used as initialization in the GD algorithm
            theta_positive --- this is the initial angle created by arctan2(P_y, P_z)
            list_of_angles ------ this is the output of go_around_unit_circle()
            list_of_distances --- list of distances between original point P and potential
                                    initial points on the swiss roll corresponding to
                                    list_of_angles
            idx ----------------- the index of the angle in list_of_angles that also
                                    indexes the smallest distance in list_of_distances
            optimal_angle ------- the angle list_of_angles[idx], i.e. this is the angle we
                                    use to find the optimal P_bar_init. This angle is 
                                    inputted into siwss_roll_parametrization()

    '''

    # Unpack the given point
    P_x, P_y, P_z = P

    # first, find the angle corresponding to point P
    theta_positive = find_angles_swiss_roll(P)

    # find a list of numbers corresponding to theta_positive within the interval [0, N * pi] of the swiss roll
    list_of_angles, remainder  = go_around_unit_circle(theta = theta_positive, theta_max=theta_max, p = np.pi, show_in_degrees = False)

    # create an empty list to store all distances later
    list_of_distances = []

    for angle in list_of_angles:
        # print('angle is ', np.degrees(angle))
        # find the point on the swiss roll corresponding to angle
        x_bar = swiss_roll_parametrization(t = angle, v = P_z, v_min=v_min, v_max=v_max)
        # print('the corresponding point on the roll is = ', x_bar)

        # compute the current distance
        cur_dist = distance_function(x = P, x_bar=x_bar)
        # print('current distance is = ', cur_dist)

        # save the current distance in a list
        list_of_distances.append(cur_dist)
        # print('list_of_distances = ', list_of_distances)
    
    # find the index of the smallest distance
    idx = np.argmin(list_of_distances)

    # find its corresponding angle in the results
    optimal_angle = list_of_angles[idx]

    # using the optimal angle, find its corresponding point on the swiss roll
    P_bar_init = swiss_roll_parametrization(t = optimal_angle, v = P_z, v_min=v_min, v_max=v_max)

    return P_bar_init, theta_positive, list_of_angles, list_of_distances, idx, optimal_angle


def generate_tangent_space_to_swiss(P_bar, theta_opt, v_opt = 0):
    '''
        Given the point P_bar on the swiss roll, this function returns
        the tangent space to the swiss roll at that point. It does so by
        \mathcal{T}_{P_bar} \mathcal {M} = span{\frac{d P_bar}{d s}, \frac{d P_bar}{d \theta}}.

        Inputs:
            P_bar ---------- ndarray, a point in 3D on the Mobius strip
            v_opt ---------- parameter of the point where we are drawing the tangent space to
            theta_opt ---------- parameter of the point where we are drawing the tangent space to
    '''


    u_1 = np.array([
        np.cos(theta_opt) - theta_opt * np.sin(theta_opt),
        np.sin(theta_opt) + theta_opt * np.cos(theta_opt),
        0
        ])
    u_2 = np.array([0, 0, 1])

    # Noralize basis vectors
    u1 = u_1 / np.linalg.norm(u_1)
    u2 = u_2 / np.linalg.norm(u_2)

    # Generate a grid of points
    grid_size = 10
    s_vals = np.linspace(-5, 5, grid_size)
    p_vals = np.linspace(-5, 5, grid_size)
    s_grid, p_grid = np.meshgrid(s_vals, p_vals)

    # Transform the grid points to lie on the plane
    x = s_grid * u1[0] + p_grid * u2[0] + P_bar[0]
    y = s_grid * u1[1] + p_grid * u2[1] + P_bar[1]
    z = s_grid * u1[2] + p_grid * u2[2] + P_bar[2]

    return x, y, z


def distance_squared(P, t, v, v_min, v_max):
    '''
        Finds the distance^2 between a point P and a point P_bar(t,v) where P_bar is found
        via the swiss roll parametrization using parameters t and v.
    '''
    P_bar = swiss_roll_parametrization(t, v, v_min=v_min, v_max=v_max)

    return 0.5*np.sum((P_bar - P)**2)


def gradient_by_hand(t, v, P, v_min, v_max):
    '''
        Computes the gradient of the distance function wrt parameters of the swiss roll.
    '''
    # normally, we want to keep v = np.clip(P[-1], v_min, v_max)
    # compute the gradient of the objective function wrt to the parameters of the swiss roll
    # P_bar_init, _, _, _, _, optimal_angle = find_P_bar_init_swiss_roll(P = P, v_min=v_min, v_max=v_max, theta_max=theta_max)

    P_next = swiss_roll_parametrization(t, v, v_min = v_min, v_max=v_max)

    # t = optimal_angle
    d_P_bar_dt = np.array([
            np.cos(t) - t * np.sin(t),
            np.sin(t) + t * np.cos(t),
            0
            ])
    d_P_bar_dv = np.array([0, 0, 1])

    d_phi_d_t = np.dot((P_next - P), d_P_bar_dt).reshape(-1,1)
    d_phi_d_v = np.dot((P_next - P), d_P_bar_dv).reshape(-1,1)

    final_grad = np.array([d_phi_d_t, d_phi_d_v]).squeeze()

    return final_grad

def project_point_swiss_roll(P,
                  learning_rate = 0.1,
                  max_iterations = 1000,
                  tolerance = 1e-6,
                  v_min = 0,
                  v_max = 10,
                  theta_max = 10 * np.pi):
    '''
        Given a point P in space, finds its projection onto the swiss roll. It does so by initializing
        gradient descent at P_bar_init. 
    '''


    P_bar_init, theta_positive, list_of_angles, list_of_distances, idx, optimal_angle = find_P_bar_init_swiss_roll(P = P,
                                                                                                                   v_min=v_min,
                                                                                                                   v_max=v_max,
                                                                                                                   theta_max=theta_max)

    # initialize parameters
    # use optimal angle to initialize t
    t = optimal_angle
    # use the last coordinate of P to initialize v
    v  = np.clip(P[-1], v_min, v_max)

    # If we want to plot the path gradient takes, we keep track of all (t,s) pairs
    # during gradient descent in a list called path
    obj_list = []

    # create a list that will collect all points. Used to plot the path of GD later
    path_params = []

    for i in range(max_iterations):

        # compute the gradient
        grad = gradient_by_hand(t, v, P, v_min, v_max)

        # update parameters
        t_new = t - learning_rate * grad[0]
        v_new = v - learning_rate * grad[1]

        # possibly unnecessary
        v_new = np.clip(v_new, v_min, v_max)

        path_params.append([t_new, v_new])

        t, v = t_new, v_new

        obj = distance_squared(P, t, v, v_min=v_min, v_max=v_max)

        obj_list.append(obj)

        # check for convergence
        if np.linalg.norm(grad) < tolerance:
            print('MET OBJ WITH t = {} and v = {} = '.format(t_new, v_new))
            print('MET OBJ VALUE AT ITERATION {}'.format(i))
            print('EXITING.....')
            break
    print("Iterations = {}".format(i))
    # compute the profjection point using the final parameters obtained by GD
    P_bar = swiss_roll_parametrization(t, v, v_min=v_min, v_max = v_max)

    return P_bar, P_bar_init, obj_list, list_of_angles, path_params, t, v

