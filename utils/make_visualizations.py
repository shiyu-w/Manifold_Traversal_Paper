import numpy as np
import plotly.graph_objects as go

# Import functions for the sphere
from utils.utils_sphere import sphere_parametrization, find_angles_sphere, find_x_bar_sphere
from utils.utils_sphere import generate_tangent_space_to_sphere, generate_normal_space_to_sphere
from utils.utils_sphere import sphere_line_intersection, choose_pt_giving_smallest_obj_value

# Import functions for the torus
from utils.utils_torus import torus_parametrization, find_x_bar_torus, find_angles_torus
from utils.utils_torus import generate_tangent_space_to_torus, generate_normal_space_to_torus

# Import functions for the mobius strip
from utils.utils_mobius import mobius_strip, project_point
from utils.utils_mobius import generate_tangent_space_to_mobius, pick_pt_with_smallest_dist

# import functions for the Swiss roll
from utils.utils_swiss_roll import swiss_roll_parametrization, project_point_swiss_roll
from utils.utils_swiss_roll import generate_tangent_space_to_swiss, distance_function



def plotly_pt_projection_onto_sphere(P, r_sphere, plot_sample_points = True,
                  plot_tangent_space = True,
                  plot_normal_space = True,
                  plot_intersection_points = True,
                  sphere_opacity = 1):
    theta, phi = find_angles_sphere(P)
    P_bar = find_x_bar_sphere(P, r_sphere)

    m = max(abs(P)) # choose the largest value, so the plot can be scaled proportionally
    m = max(m, 1//m)

    x_sphere, y_sphere, z_sphere = sphere_parametrization(r_sphere)
    x_tangent, y_tangent, z_tangent = generate_tangent_space_to_sphere(P, theta, phi, r_sphere)
    line_pts, n1 = generate_normal_space_to_sphere(P, theta, phi, t_range = np.linspace(-m//3, 2*m, 100))
    t1, t2, pt1, pt2 = sphere_line_intersection(point_on_line=P, line_direction=n1, radius=r_sphere)
    final_pt = choose_pt_giving_smallest_obj_value(P, P_bar, pt1, pt2)
    
    # Create the plot
    fig = go.Figure()

    # Add the sphere surface
    fig.add_trace(go.Surface(x=x_sphere, y=y_sphere, z=z_sphere,
                             colorscale='plotly3', showscale=False, opacity=sphere_opacity))

    if plot_sample_points:
        # Add the scatter plot of sample points
        fig.add_trace(go.Scatter3d(x=np.array([P[0]]), y=np.array([P[1]]), z=np.array([P[2]]),
                                mode='markers', marker=dict(size=10, color='turquoise', showscale=False, opacity=1),
                                name='point P'
                                ))
        fig.add_trace(go.Scatter3d(x=np.array([P_bar[0]]), y=np.array([P_bar[1]]), z=np.array([P_bar[2]]),
                                mode='markers', marker=dict(size=14, symbol='diamond', color='green', showscale=False, opacity=1),
                                name='point P_bar, true projection'
                                ))
        fig.add_trace(go.Scatter3d(x=np.array([final_pt[0]]), y=np.array([final_pt[1]]), z=np.array([final_pt[2]]),
                                mode='markers', marker=dict(size=9, symbol='circle', color='red', showscale=False, opacity=1),
                                name='intersection point that gives smaller obj value'
                                ))
        
    new_array_x = np.array([pt1[0], pt2[0]])

    if plot_intersection_points:
        fig.add_trace(go.Scatter3d(x=new_array_x, y=np.array([pt1[1], pt2[1]]), z=np.array([pt1[2], pt2[2]]),
                                   mode='markers', marker=dict(size=15, color='purple',
                                    symbol='cross', showscale=False, opacity=1),
                                    name='Intersection points'))
    
    if plot_tangent_space:
        # Create the tangent space
        fig.add_trace(go.Surface(x=x_tangent, y=y_tangent, z=z_tangent,
                                    colorscale=[[0, 'blue'], [1, 'blue']],  # Single color
                                    showscale=False, opacity=0.7))

    if plot_normal_space:
        fig.add_trace(go.Scatter3d(x=line_pts[:, 0], y=line_pts[:, 1], z=line_pts[:, 2],
                                mode='lines', line=dict(color='blue', width=6),
                                name='Normal space'))
    
    # Update layout
    fig.update_layout(
        # title='Sphere',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data',
            aspectratio=dict(x=1, y=1, z=1)),
            width=800,
            height=800)

    # Show the plot
    fig.show()

    return theta, phi, P_bar, pt1, pt2




def plotly_pt_projection_onto_torus(P = np.array([7, 7, 7]), R = 5, r = 2,
                               plot_tangent_space = True,
                               plot_sample_points = True,
                               plot_normal_space = True,
                               plot_line_btw_pts = True,
                               manifold_opacity = 0.2,
                               tangent_space_opacity = 0.3):

    # Given a point P, find P_bar on the torus
    P_bar = find_x_bar_torus(P)
    
    # find angles
    theta, phi = find_angles_torus(P)

    # Generate torus surface
    x_torus, y_torus, z_torus = torus_parametrization(R, r)

    # Generate tangent space at P_bar
    x_tangent, y_tangent, z_tangent = generate_tangent_space_to_torus(P_bar, theta, phi)
    line_pts, n1 = generate_normal_space_to_torus(P, theta, phi, t_range = np.linspace(-5,5,100))

    # Create the plot
    fig = go.Figure()

    # Add the torus surface
    fig.add_trace(go.Surface(x=x_torus, y=y_torus, z=z_torus, colorscale='plotly3', showscale=False, opacity=manifold_opacity))

    if plot_sample_points:
        # Add the scatter plot of sample points
        fig.add_trace(go.Scatter3d(x=np.array([P[0]]), y=np.array([P[1]]), z=np.array([P[2]]),
                                mode='markers', marker=dict(size=10, color='turquoise', showscale=False, opacity=0.9),
                                name='point P'
                                ))
        fig.add_trace(go.Scatter3d(x=np.array([P_bar[0]]), y=np.array([P_bar[1]]), z=np.array([P_bar[2]]),
                                mode='markers', marker=dict(size=10, symbol='diamond', color='red', showscale=False, opacity=0.9),
                                name='point P_bar, true projection'
                                ))

    if plot_tangent_space:
        # Create the tangent space
        fig.add_trace(go.Surface(x=x_tangent, y=y_tangent, z=z_tangent,
                                    colorscale=[[0, 'blue'], [1, 'blue']],  # Single color
                                    showscale=False, opacity=tangent_space_opacity))
    if plot_normal_space:
        fig.add_trace(go.Scatter3d(x=line_pts[:, 0], y=line_pts[:, 1], z=line_pts[:, 2],
                                mode='lines', line=dict(color='blue', width=6),
                                name='Normal space'))

    if plot_line_btw_pts:
        # Plot a line between the points
        fig.add_trace(go.Scatter3d(x=[P[0], P_bar[0]], 
                        y=[P[1], P_bar[1]], 
                        z=[P[2], P_bar[2]],
                        mode='lines',
                        line=dict(color='turquoise', width=5),
                        marker=dict(size=5, color=['green', 'yellow'], showscale=False),
                        name='A line connecting the pts'
                        ))

    # Update layout
    fig.update_layout(
        # title='Torus',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data',
            aspectratio=dict(x=1, y=1, z=1)),
        width=800,
        height=800,
    )

    # Show the plot
    fig.show()

    return theta, phi, P_bar




def plotly_pt_projection_onto_sphere_and_torus(P,
                                               plot_sample_points = True,
                                               plot_line_btw_pts  = True,
                                               plot_tangent_space = True,
                                               tangent_space_opacity = 0.8,
                                               sphere_opacity = 0.3,
                                               torus_opacity = 0.3,
                                               R_torus = 5,
                                               r_torus = 2,
                                               r_sphere = 1):
    P_bar_sphere = find_x_bar_sphere(P, r_sphere=r_sphere)
    P_bar_torus = find_x_bar_torus(P, R=R_torus, r=r_torus)

    distance_to_sphere = np.linalg.norm(P-P_bar_sphere)
    distance_to_torus = np.linalg.norm(P-P_bar_torus)

    if distance_to_sphere <= distance_to_torus:
        print("THE SPHERE IS CLOSER")
        print("PROJECTING ONTO THE SPHERE...")
        final_P_bar = P_bar_sphere
    else:
        print("THE TORUS IS CLOSER")
        print("PROJECTING ONTO THE TORUS...")
        final_P_bar = P_bar_torus

    # Make plots
    theta_sphere, phi_sphere = find_angles_sphere(P)
    theta_torus, phi_torus   = find_angles_torus(P)
    

    # Generate sphere surface
    x_sphere, y_sphere, z_sphere = sphere_parametrization(r_sphere)
    x_tangent_sphere, y_tangent_sphere, z_tangent_sphere = generate_tangent_space_to_sphere(P, theta_sphere, phi_sphere,r_sphere=r_sphere)

    # Generate torus surface
    x_torus, y_torus, z_torus = torus_parametrization(R_torus, r_torus)
    x_tangent_torus, y_tangent_torus, z_tangent_torus = generate_tangent_space_to_torus(P_bar_torus, theta_torus, phi_torus)

    # Create the plot
    fig = go.Figure()

    # Plot the sphere surface
    fig.add_trace(go.Surface(x=x_sphere, y=y_sphere, z=z_sphere, colorscale='plotly3', showscale=False, opacity=sphere_opacity))

    # Plot the torus surface
    fig.add_trace(go.Surface(x=x_torus, y=y_torus, z=z_torus, colorscale='plotly3', showscale=False, opacity=torus_opacity))


    if plot_sample_points:
        # Add the scatter plot of sample points
        fig.add_trace(go.Scatter3d(x=np.array([P[0]]), y=np.array([P[1]]), z=np.array([P[2]]),
                                mode='markers', marker=dict(size=13, color='purple', showscale=False, opacity=0.9),
                                name='point P'
                                ))
        fig.add_trace(go.Scatter3d(x=np.array([P_bar_torus[0]]), y=np.array([P_bar_torus[1]]), z=np.array([P_bar_torus[2]]),
                                mode='markers', marker=dict(size=13, color='turquoise', showscale=False, opacity=0.9),
                                name='point P_bar_torus'
                                ))
        fig.add_trace(go.Scatter3d(x=np.array([P_bar_sphere[0]]), y=np.array([P_bar_sphere[1]]), z=np.array([P_bar_sphere[2]]),
                                mode='markers', marker=dict(size=13, color='turquoise', showscale=False, opacity=0.9),
                                name='point P_bar_sphere'
                                ))
        fig.add_trace(go.Scatter3d(x=np.array([final_P_bar[0]]), y=np.array([final_P_bar[1]]), z=np.array([final_P_bar[2]]),
                                mode='markers', marker=dict(size=8, symbol='diamond', color='red', showscale=False, opacity=0.9),
                                name='point P_bar, true projection'
                                ))
        

    if plot_tangent_space:
        # Create the tangent space
        fig.add_trace(go.Surface(x=x_tangent_sphere, y=y_tangent_sphere, z=z_tangent_sphere,
                                    colorscale=[[0, 'blue'], [1, 'blue']],  # Single color
                                    showscale=False, opacity=tangent_space_opacity))
        # Create the tangent space
        fig.add_trace(go.Surface(x=x_tangent_torus, y=y_tangent_torus, z=z_tangent_torus,
                                    colorscale=[[0, 'blue'], [1, 'blue']],  # Single color
                                    showscale=False, opacity=tangent_space_opacity))

    if plot_line_btw_pts:
        # Plot a line between the points
        fig.add_trace(go.Scatter3d(x=[P[0], P_bar_torus[0]], 
                        y=[P[1], P_bar_torus[1]], 
                        z=[P[2], P_bar_torus[2]],
                        mode='lines',
                        line=dict(color='turquoise', width=5),
                        marker=dict(size=5, color=['green', 'yellow'], showscale=False),
                        name='line between P and P_bar_torus'
                        ))
        # Plot a line between the points
        fig.add_trace(go.Scatter3d(x=[P[0], P_bar_sphere[0]], 
                        y=[P[1], P_bar_sphere[1]], 
                        z=[P[2], P_bar_sphere[2]],
                        mode='lines',
                        line=dict(color='turquoise', width=5),
                        marker=dict(size=5, color=['green', 'yellow'], showscale=False),
                        name='line between P and P_bar_sphere'
                        ))
            
    # Update layout
    fig.update_layout(
        # title='Sphere',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data',
            aspectratio=dict(x=1, y=1, z=1)),
            width=800,
            height=800)

    # Show the plot
    fig.show()

    return {'theta_sphere': theta_sphere, 'phi_sphere': phi_sphere, 'theta_torus': theta_torus,\
             "phi_torus": phi_torus, 'distance_to_sphere': distance_to_sphere, 'distance_to_torus': distance_to_torus, \
                'P_bar_sphere': P_bar_sphere, 'P_bar_torus': P_bar_torus, 'final_P_bar': final_P_bar}


def return_only_P_final_sphere_and_torus(P, R_torus = 5, r_torus = 2, r_sphere = 1):
    P_bar_sphere = find_x_bar_sphere(P, r_sphere=r_sphere)
    P_bar_torus = find_x_bar_torus(P, R=R_torus, r=r_torus)

    distance_to_sphere = np.linalg.norm(P-P_bar_sphere)
    distance_to_torus = np.linalg.norm(P-P_bar_torus)

    if distance_to_sphere <= distance_to_torus:
        # print("THE SPHERE IS CLOSER")
        # print("PROJECTING ONTO THE SPHERE...")
        final_P_bar = P_bar_sphere
    else:
        # print("THE TORUS IS CLOSER")
        # print("PROJECTING ONTO THE TORUS...")
        final_P_bar = P_bar_torus

    return final_P_bar


def plot_mobius_visualization(P,
                                    optimal_point_params,
                                    unique_sublists,
                                    list_of_local_min_initializations,
                                    list_of_regular_initializations,
                                    path_lm1,
                                    path_lm2,
                                    path_normal,
                                    plot_local_min_initializations = True,
                                    plot_regular_initializations = False,
                                    plot_line_btw_optimal_pts = True,
                                    plot_line_btw_suboptimal_pts = True,
                                    plot_path = True,
                                    plot_path_lm1 = True,
                                    plot_path_lm2 = True,
                                    plot_path_normal = True,
                                    mobius_opacity = 0.3,
                                    plot_sample_points = True,
                                    plot_unique_sublist = True,
                                    R = 1,
                                    use_momentum = False,
                                    dec = 3):

    # Generate data for the Möbius strip
    t_grid = np.linspace(0, 2*np.pi, 100)
    s_grid = np.linspace(-1, 1, 25)
    t_grid, s_grid = np.meshgrid(t_grid, s_grid)

    x_mobius, y_mobius, z_mobius = mobius_strip(s_grid, t_grid)
    P_bar = mobius_strip(s = optimal_point_params[0], t = optimal_point_params[1])
    unique_list_of_final_pts_on_mobius = np.array([mobius_strip(s, t, R) for s, t in unique_sublists])
    unique_list_of_final_pts_on_mobius = np.round(unique_list_of_final_pts_on_mobius, dec)

    # Create the plot
    fig = go.Figure()

    # Plot the Möbius strip surface
    mobius_surface = go.Surface(x=x_mobius, y=y_mobius, z=z_mobius, colorscale='plotly3', showscale=False, opacity=mobius_opacity)
    fig = go.Figure(data=[mobius_surface])

    # Plot the original point and its projection
    if plot_sample_points:
        original_point_P = go.Scatter3d(x=np.array([P[0]]), y=np.array([P[1]]), z=np.array([P[2]]),
                                mode='markers',
                                marker=dict(size=10, color='purple', showscale=False, opacity=1),
                                name='point P'
                                )
        fig.add_trace(original_point_P)


        # plot the optimal projection
        # P_bar = mobius_strip(s = optimal_point_params[0], t = optimal_point_params[1])
        projected_point_P_bar = go.Scatter3d(x=np.array([P_bar[0]]), y=np.array([P_bar[1]]), z=np.array([P_bar[2]]),
                                mode='markers',
                                marker=dict(size=14, symbol='diamond', color='green', showscale=False, opacity=1),
                                name='point P_bar, true projection'
                                )
        fig.add_trace(projected_point_P_bar)

        # Create scatter plot for the gradient descent path
        # below are the local mins and global mins

    # unique_list_of_final_pts_on_mobius = np.array([mobius_strip(s, t, R) for s, t in unique_sublists])
    # unique_list_of_final_pts_on_mobius = np.round(unique_list_of_final_pts_on_mobius, dec)
    
    if plot_unique_sublist:
        fig.add_trace(go.Scatter3d(x=unique_list_of_final_pts_on_mobius[:, 0], y=unique_list_of_final_pts_on_mobius[:, 1], z=unique_list_of_final_pts_on_mobius[:, 2],
                                mode='markers',
                                marker=dict(size=7, symbol='x', color='red', showscale=False, opacity=1),
                                name='unique_list_of_final_pts_on_mobius'
                                ))

    if plot_local_min_initializations:
        # find all candidate points in 3D with (x,y,z) coordinates
        list_of_init_pts = []
        for param in list_of_local_min_initializations:
                pt = mobius_strip(param[0], param[1], R)
                list_of_init_pts.append(pt)
        # plot all candidate plots
        for pt in list_of_init_pts:
            fig.add_trace(go.Scatter3d(x=np.array([pt[0]]), y=np.array([pt[1]]), z=np.array([pt[2]]),
                                    mode='markers',
                                    marker=dict(size=8, symbol='circle', color='black', showscale=False, opacity=1),
                                    name='list_of_init_pts'
                                    ))
            
    if plot_regular_initializations:
        # find all candidate points in 3D with (x,y,z) coordinates
        list_of_regular_init_pts = []
        for param in list_of_regular_initializations:
                pt = mobius_strip(param[0], param[1], R)
                list_of_regular_init_pts.append(pt)
        # plot all candidate plots
        for pt in list_of_regular_init_pts:
            fig.add_trace(go.Scatter3d(x=np.array([pt[0]]), y=np.array([pt[1]]), z=np.array([pt[2]]),
                                    mode='markers',
                                    marker=dict(size=3, symbol='circle', color='blue', showscale=False, opacity=1),
                                    name='list_of_init_pts'
                                    ))

    # Plot a line between the points
    if plot_line_btw_optimal_pts:
        line_plot = go.Scatter3d(x=[P[0], P_bar[0]], 
                            y=[P[1], P_bar[1]], 
                            z=[P[2], P_bar[2]],
                            mode='lines',
                            line=dict(color='turquoise', width=5),
                            marker=dict(size=5, color=['green', 'yellow'], showscale=False),
                            name='line between P and optimal P_bar'
                            )
        fig.add_trace(line_plot)

    # Plot a line between the all local/global minima and the given
    if plot_line_btw_suboptimal_pts:
        for pt in unique_list_of_final_pts_on_mobius:
            line_plot = go.Scatter3d(x=[P[0], pt[0]], 
                                y=[P[1], pt[1]], 
                                z=[P[2], pt[2]],
                                mode='lines',
                                line=dict(color='turquoise', width=5),
                                marker=dict(size=5, color=['green', 'yellow'], showscale=False),
                                name='line between P and P_bar'
                                )
            fig.add_trace(line_plot)


    # add the GD path to the plot
    if plot_path:
        if plot_path_lm1:
            # Create scatter plot for the gradient descent path
            path_lm1_points_on_manifold = np.array([mobius_strip(s, t, R) for s, t in path_lm1])

            fig.add_trace(go.Scatter3d(x = path_lm1_points_on_manifold[:, 0],
                                    y = path_lm1_points_on_manifold[:, 1],
                                    z = path_lm1_points_on_manifold[:, 2],
                                    mode='lines+markers', marker=dict(size=2, color='blue'),
                                    line=dict(color='green', width=2), name='Gradient Descent Path'))
        if plot_path_lm2:
            # Create scatter plot for the gradient descent path
            path_lm2_points_on_manifold = np.array([mobius_strip(s, t, R) for s, t in path_lm2])

            fig.add_trace(go.Scatter3d(x = path_lm2_points_on_manifold[:, 0],
                                    y = path_lm2_points_on_manifold[:, 1],
                                    z = path_lm2_points_on_manifold[:, 2],
                                    mode='lines+markers', marker=dict(size=2, color='blue'),
                                    line=dict(color='green', width=2), name='Gradient Descent Path'))
        if plot_path_normal:
            # Create scatter plot for the gradient descent path
            path_normal_points_on_manifold = np.array([mobius_strip(s, t, R) for s, t in path_normal])

            fig.add_trace(go.Scatter3d(x = path_normal_points_on_manifold[:, 0],
                                    y = path_normal_points_on_manifold[:, 1],
                                    z = path_normal_points_on_manifold[:, 2],
                                    mode='lines+markers', marker=dict(size=2, color='blue'),
                                    line=dict(color='green', width=2), name='Gradient Descent Path'))
        

    # Update layout
    fig.update_layout(
        title='Projection onto Möbius Strip. Using momentum = {}'.format(use_momentum),
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data',
            aspectratio=dict(x=1, y=1, z=1)),
            width=1000,
            height=1000)

    # Show the plot
    fig.show()
    # path is a list of (s,t) pairs
    # path_points_on_manifold is a collection of (x,y,z) pairs on the manifold

    # return path_points_on_manifold, path, grad_list, obj_list
    return unique_list_of_final_pts_on_mobius, P_bar


def plotly_pt_projection_onto_mobius_strip(point, dec = 3, num_samples = 100, num_local_mins = 50, theta_offset = 4e-7,
                                           s_offset = 1e-6, learning_rate = 0.01, max_iterations = 1000, tolerance = 1e-6, R = 1,
                                    plot_local_min_initializations = True, # black dots
                                    plot_regular_initializations = False,
                                    plot_line_btw_optimal_pts = True,
                                    plot_line_btw_suboptimal_pts = True,
                                    plot_path_lm1 = True,
                                    plot_path_lm2 = True,
                                    plot_path_normal = True,
                                    mobius_opacity = 0.3,
                                    plot_sample_points = True,
                                    plot_unique_sublist = True):    
    
    optimal_point_params, unique_sublists, final_list_of_params, \
        list_of_local_min_initializations, list_of_regular_initializations,\
            theta, theta_2, [path_lm1, path_lm2, path_normal] = pick_pt_with_smallest_dist(P = point, 
                                                                                dec = dec,
                                                                                num_samples = num_samples,
                                                                                num_local_mins = num_local_mins,
                                                                                theta_offset = theta_offset, s_offset = s_offset,
                                                                                learning_rate = learning_rate, max_iterations = max_iterations,
                                                                                tolerance = tolerance, R = R)
    
    unique_list_of_final_pts_on_mobius, P_bar = plot_mobius_visualization(P = point,
                                    optimal_point_params = optimal_point_params,
                                    unique_sublists = unique_sublists,
                                    list_of_local_min_initializations = list_of_local_min_initializations,
                                    list_of_regular_initializations = list_of_regular_initializations,
                                    path_lm1 = path_lm1[-1], # is a list of lists
                                    path_lm2 = path_lm2[-1],
                                    path_normal = path_normal[-1],
                                    plot_local_min_initializations = plot_local_min_initializations, # black dots
                                    plot_regular_initializations = plot_regular_initializations,
                                    plot_line_btw_optimal_pts = plot_line_btw_optimal_pts,
                                    plot_line_btw_suboptimal_pts = plot_line_btw_suboptimal_pts,
                                    plot_path_lm1 = plot_path_lm1,
                                    plot_path_lm2 = plot_path_lm2,
                                    plot_path_normal = plot_path_normal,
                                    mobius_opacity = mobius_opacity,
                                    plot_sample_points = plot_sample_points,
                                    plot_unique_sublist = plot_unique_sublist,
                                    dec = dec)
    
    return unique_list_of_final_pts_on_mobius, theta, theta_2, P_bar

def return_only_P_final_mobius(P, dec = 3, num_samples = 100, num_local_mins = 50, theta_offset = 4e-7,
                               s_offset = 1e-6, learning_rate = 0.01, max_iterations = 1000, tolerance = 1e-6, R = 1):
    
    optimal_point_params, unique_sublists, final_list_of_params, \
        list_of_local_min_initializations, list_of_regular_initializations,\
            theta, theta_2, [path_lm1, path_lm2, path_normal] = pick_pt_with_smallest_dist(P = P, 
                                                                                dec = dec,
                                                                                num_samples = num_samples,
                                                                                num_local_mins = num_local_mins,
                                                                                theta_offset = theta_offset, s_offset = s_offset,
                                                                                learning_rate = learning_rate, max_iterations = max_iterations,
                                                                                tolerance = tolerance, R = R)
    
    
    P_bar = mobius_strip(s = optimal_point_params[0], t = optimal_point_params[1])
    return P_bar


def plotly_pt_projection_onto_swiss_roll(P,
                                     roll_opacity = 1,
                                     learning_rate = 0.01,
                                     max_iterations = 1000,
                                     tolerance = 1e-6,
                                     plot_original_point = True,
                                     plot_candidate_initializations = True,
                                     plot_initial_proj = True,
                                     plot_projected_point = True,
                                     plot_edge_point = True,
                                     plot_line_btw_P_P_bar = True,
                                     plot_line_btw_P_P_bar_init = True,
                                     plot_line_btw_P_edge_pt = True,
                                     plot_tangent_space = True,
                                     plot_path = True,
                                     colorscale='plotly3',
                                     tangent_space_opacity = 0.5,
                                     theta_max = 5 * np.pi,
                                     v_max = 20,
                                     v_min = -20,
                                     N_pts = 100):
    

    # P_bar, theta_positive, list_of_angles, list_of_distances, idx, optimal_angle = find_P_bar_init_swiss_roll(P = P,
                                                                                        # v_min=v_min, v_max=v_max)
    
    P_bar, P_bar_init, obj_list, list_of_angles, path_params, t_final, v_final = project_point_swiss_roll(P, learning_rate = learning_rate,
                                                                                          max_iterations = max_iterations,
                                                                                          tolerance = tolerance,
                                                                                          v_min = v_min,
                                                                                          v_max = v_max,
                                                                                          theta_max = theta_max)

    # find the edge point of the roll just in case it's closer to the original point
    edge_point = swiss_roll_parametrization(t = theta_max, v = P[-1], v_min=v_min, v_max=v_max)
    d_edge_P = distance_function(x=P, x_bar=edge_point)
    d_P_P_bar = distance_function(x = P, x_bar=P_bar)

    print('INITIAL DISTANCE: ||P - P_bar_init||^2 = ', distance_function(x = P, x_bar=P_bar_init))
    print('FINAL DISTANCE:   ||P - P_bar     ||^2 = ', d_P_P_bar)
    print('EDGE DISTANCE:    ||P - edge_pt   ||^2 = ', d_edge_P)

    if d_edge_P <= d_P_P_bar:
        P_final = edge_point
    else:
        P_final = P_bar

    # Generate data for the Swiss roll surface
    t    = np.linspace(0, theta_max, N_pts)
    v    = np.linspace(v_min, v_max, N_pts)
    t, v = np.meshgrid(t, v)

    # generate the swiss roll
    swiss_roll_array = swiss_roll_parametrization(t, v, v_min=v_min, v_max=v_max)

    x_swiss, y_swiss, z_swiss = swiss_roll_array[:,:,0], swiss_roll_array[:,:,1], swiss_roll_array[:,:,2]

    # Create the plot
    fig = go.Figure()

    # Add the Swiss roll surface
    fig.add_trace(go.Surface(x=x_swiss, y=y_swiss, z=z_swiss, colorscale=colorscale, showscale=False, opacity=roll_opacity))

    if plot_original_point:
        original_point_P = go.Scatter3d(x=np.array([P[0]]), y=np.array([P[1]]), z=np.array([P[2]]),
                                mode='markers', marker=dict(size=8, color='purple', showscale=False, opacity=1),
                                name='original point P'
                                )
        fig.add_trace(original_point_P)

    if plot_candidate_initializations:
        # find all candidate points
        list_of_pts = []
        for angle in list_of_angles:
            pt = swiss_roll_parametrization(t = angle, v = P[-1], v_min=v_min, v_max = v_max)
            list_of_pts.append(pt)

        # plot all candidate plots
        for p in list_of_pts:
            fig.add_trace(go.Scatter3d(x=np.array([p[0]]), y=np.array([p[1]]), z=np.array([p[2]]),
                                mode='markers', marker=dict(size=5, symbol='x', color='blue', showscale=False, opacity=1),
                                name='potential initial point for GD'
                                ))

    if plot_initial_proj:
        point_P_bar_init = go.Scatter3d(x=np.array([P_bar_init[0]]), y=np.array([P_bar_init[1]]), z=np.array([P_bar_init[2]]),
                                    mode='markers', marker=dict(size=8, symbol='cross', color='turquoise', showscale=False, opacity=1),
                                    name='point P_bar_init, GD is initialized here'
                                    )
        fig.add_trace(point_P_bar_init)

    if plot_projected_point:
        projected_point_P_bar = go.Scatter3d(x=np.array([P_bar[0]]), y=np.array([P_bar[1]]), z=np.array([P_bar[2]]),
                                    mode='markers', marker=dict(size=8, symbol='diamond', color='red', showscale=False, opacity=1),
                                    name='point P_bar, final pt achieved by GD'
                                    )
        fig.add_trace(projected_point_P_bar)
        


    if plot_edge_point:
        edge_point_plot = go.Scatter3d(x=np.array([edge_point[0]]), y=np.array([edge_point[1]]), z=np.array([edge_point[2]]),
                                mode='markers', marker=dict(size=8, color='green', showscale=False, opacity=1),
                                name='edge point closest to P'
                                )
        fig.add_trace(edge_point_plot)



    # Plot a line between the points
    if plot_line_btw_P_P_bar:
        line_plot = go.Scatter3d(x=[P[0], P_bar[0]], 
                        y=[P[1], P_bar[1]], 
                        z=[P[2], P_bar[2]],
                        mode='lines',
                        line=dict(color='red', width=5),
                        marker=dict(size=5, color=['green', 'yellow'], showscale=False),
                        name='line between P and P_bar'
                        )
        fig.add_trace(line_plot)

    # Plot a line between the points
    if plot_line_btw_P_P_bar_init:
        line_plot = go.Scatter3d(x=[P[0], P_bar_init[0]], 
                        y=[P[1], P_bar_init[1]], 
                        z=[P[2], P_bar_init[2]],
                        mode='lines',
                        line=dict(color='turquoise', width=5),
                        marker=dict(size=5, color=['red', 'yellow'], showscale=False),
                        name='line between P and P_bar_init'
                        )
        fig.add_trace(line_plot)

    # Plot a line between the points
    if plot_line_btw_P_edge_pt:
        line_plot = go.Scatter3d(x=[P[0], edge_point[0]], 
                        y=[P[1], edge_point[1]], 
                        z=[P[2], edge_point[2]],
                        mode='lines',
                        line=dict(color='green', width=5),
                        marker=dict(size=5, color=['red', 'yellow'], showscale=False),
                        name='line between P and edge_point'
                        )
        fig.add_trace(line_plot)
    
    # add the GD path to the plot
    if plot_path:
        # Create scatter plot for the gradient descent path
        path_points_on_manifold = np.array([swiss_roll_parametrization(t_, v_, v_min=v_min, v_max=v_max) for t_, v_ in path_params])
        fig.add_trace(go.Scatter3d(x=path_points_on_manifold[:, 0],
                                y=path_points_on_manifold[:, 1],
                                z=path_points_on_manifold[:, 2],
                                mode='lines+markers', marker=dict(size=2, color='blue'),
                                line=dict(color='green', width=2), name='Gradient Descent Path'))


    if plot_tangent_space:
        # compute the tangent space
        x_tangent, y_tangent, z_tangent = generate_tangent_space_to_swiss(P_final, theta_opt = t_final)
        # Plot the tangent space
        fig.add_trace(go.Surface(x=x_tangent, y=y_tangent, z=z_tangent,
                                        # colorscale=[[0, 'blue'], [1, 'blue']],  # Single color
                                        colorscale = 'Reds',
                                        showscale=False, opacity=tangent_space_opacity))

    # Update layout
    fig.update_layout(
        title='Swiss Roll',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data',
            aspectratio=dict(x=1, y=1, z=1)),
        width=900,
        height=900,
    )

    # Show the plot
    fig.show()

    return obj_list, P_final, P_bar, P_bar_init, edge_point

def return_only_P_final_swiss_roll(P, learning_rate = 0.01, max_iterations = 1000, tolerance = 1e-6, theta_max = 5 * np.pi,
                                   v_max = 20,
                                   v_min = -20):
    
    '''
        This function is the oracle denoiser for the swiss roll. It takes a point P and returns its projection P_final.
        This function is identical to the first part of plotly_pt_projection_onto_swiss_roll() but does not do any
        plotting.
    '''
    
    P_bar, P_bar_init, obj_list, list_of_angles, path_params, t_final, v_final = project_point_swiss_roll(P, learning_rate = learning_rate,
                                                                                          max_iterations = max_iterations,
                                                                                          tolerance = tolerance,
                                                                                          v_min = v_min,
                                                                                          v_max = v_max,
                                                                                          theta_max = theta_max)

    # find the edge point of the roll just in case it's closer to the original point
    edge_point = swiss_roll_parametrization(t = theta_max, v = P[-1], v_min=v_min, v_max=v_max)
    d_edge_P = distance_function(x=P, x_bar=edge_point)
    d_P_P_bar = distance_function(x = P, x_bar=P_bar)

    if d_edge_P <= d_P_P_bar:
        P_final = edge_point
    else:
        P_final = P_bar

    return P_final