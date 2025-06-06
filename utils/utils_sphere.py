
import numpy as np
import plotly.graph_objects as go


def sphere_parametrization(r_sphere=1, n=100):
    '''
        This function plots a sphere with radius R in 3D.
    '''

    theta = np.linspace(0, 2*np.pi, n)
    phi = np.linspace(0, 2*np.pi, n)
    theta, phi = np.meshgrid(theta, phi)

    x = r_sphere*np.cos(theta)*np.sin(phi)
    y = r_sphere*np.sin(theta)*np.sin(phi)
    z = r_sphere*np.cos(phi)

    return x, y, z

def find_x_bar_sphere(p, r_sphere):
    '''
        This function finds the projection of a point $p \in \R^3$ onto the sphere by
        normalizing it using the formula:
            Proj_{\mathbb{S}^{D-1}} [p] = p / ||p||_2

        Inputs:
            p ------- the given point, a 3-dim numpy array

        Outputs:
            p_bar --- projection of point p onto the sphere, a 3-dimensional numpy array

    '''
    x_P, y_P, z_P = p
    P_norm = np.linalg.norm(p)
    p_bar = r_sphere*np.array([x_P / P_norm, y_P / P_norm, z_P / P_norm])

    return p_bar

def find_angles_sphere(P):
    '''
    Given a point P in 3D, find the angles theta and phi to represent it
    in spherical coordinates. This will be used to find the angles
    and use them in other functions (e.g. tangent_space_sphere()). Note
    that angles theta and phi do not depend on the radius of the sphere.

    Returns angles in radians.
    '''
    # Unpack the given point
    x_P, y_P, z_P = P
    
    # Calculate rho (distance from origin to point)
    rho = np.sqrt(x_P**2 + y_P**2 + z_P**2)
    
    # Calculate phi (inclination angle from positive z-axis)
    phi = np.arccos(z_P / rho)
    
    # Calculate theta (azimuthal angle in x-y plane from positive x-axis)
    # arctan2 chooses the quadrant correctly, is only defined for reals
    theta = np.arctan2(y_P, x_P)
    
    # Ensure theta is in [0, 2π)
    theta = theta % (2 * np.pi)

    return theta, phi

def generate_tangent_space_to_sphere(P, theta, phi, r_sphere):
    '''
        Given a point P not on the sphere, this function finds the tangent
        space at point P_bar, i.e. at the projection of P onto the sphere.
        It finds the tangent space by returning grid points x, y, z, which
        can later be plotted in plotly.

        Inputs:
            P --------- a given point, a 3-dim numpy array
            theta ----- parameter of spherical coordinates given point P
            phi ------- parameter of spherical coordinates given point P
        Outputs:
            x, y, z --- grid points of the tangent space to lie on a plane
    '''

    P_bar = find_x_bar_sphere(P, r_sphere)
    
    # Find tangent space basis vectors
    u_1 = np.array([-np.sin(theta), np.cos(theta), 0])
    u_2 = np.array([np.cos(theta)*np.cos(phi), np.sin(theta)*np.cos(phi), -np.sin(phi)])

    # Normalize them, just incase
    u1 = u_1 / np.linalg.norm(u_1)
    u2 = u_2 / np.linalg.norm(u_2)

    # Generate a grid of points to lie on the tangent plane
    grid_size = 10
    s = np.linspace(-1,1,grid_size)
    t = np.linspace(-1,1,grid_size)
    s, t = np.meshgrid(s, t)

    # Transform the grid points to lie on the plane, then shift it to be centered at P_bar
    x = s*u1[0] + t*u2[0] + P_bar[0]
    y = s*u1[1] + t*u2[1] + P_bar[1]
    z = s*u1[2] + t*u2[2] + P_bar[2]

    return x, y, z


def generate_normal_space_to_sphere(point, theta, phi, t_range = np.linspace(-2,2,100)):
    """
        This function generates points along a vector. We use this to plot the normal
        space which, in the case of the sphere and the torus in 3D, is a line.
            line = point + t_tange * direction_vector
        
        Inputs:
            point (array-like) --------- A point on the line (x, y, z).
            vector (array-like) -------- The direction vector of the line (a, b, c).
            t_range (array-like) ------- Range of parameter t to generate points.
        
        Outputs:
            line_pts ------------------- Array of points on the line.
    """

    # Define the basis vectors of the tangent space
    u_1 = np.array([-np.sin(theta), np.cos(theta), 0])
    u_2 = np.array([np.cos(theta)*np.cos(phi), np.sin(theta)*np.cos(phi), -np.sin(phi)])

    # Normalize them
    u1 = u_1 / np.linalg.norm(u_1)
    u2 = u_2 / np.linalg.norm(u_2)

    # find their cross product to find the normal vector to the tangent space
    n1 = np.cross(u1, u2) # only works for 3D

    # Normalize the direction vector
    n1 = n1 / np.linalg.norm(n1)

    # Initialize an empty list to store the points
    points = []
    
    # Generate points using a for loop
    for t in t_range:
        new_point = point + t * n1
        points.append(new_point)

    # convert to an array
    line_pts = np.array(points)
    
    return line_pts, n1

def sphere_line_intersection(line_direction = np.array([0,0,1]),
                             center=np.array([0,0,0]),
                             radius=1,
                             point_on_line = np.array([0,0,0])):
    '''
        This function finds the intersection of a line and a sphere.
            vector notation of the sphere: ||x_vec - c_vec||^2_2 = r^2 (1)
            equation of a line starting at o_vec: x_vec = o_vec + t n_vec (2)
            where x_vec is a point on the sphere and the line, c_vec is the center of the sphere,
            r is the radius of the sphere, o_vec is the origin of the line, t is the distance from
            the origin of the line, and n_vec is the direction of the line. Finding the intersection
            of the line and the sphere is equivalent to solving for scalar t. We substitute x_vec in
            (1) with x_vec in (2) and solve for the scalar t.

        Inputs:
            line_direction ------ a 3-dimensional vector defining the direction of the line
            center -------------- in R^3, the center of the sphere, is the origin by default
            radius -------------- radius of the sphere, 1 by default
            point_on_line ------- a point on the line

        Outputs:
            case 1 ------ returns two zeros and two empty numpy arrays if there is no intersection
            case 2 ------ returns the scalar value t and 0, the single point of intersection and an
                            empty numpy array
            case 3 ------ returns scalar values t1 and t2, as well as the two points of intersection 
    '''

    # Normalize the line direction vector
    line_direction = line_direction / np.linalg.norm(line_direction)
    
    # Calculate the coefficients of the quadratic equation. These are scalars.
    a = np.dot(line_direction, line_direction)
    b = 2 * np.dot(line_direction, point_on_line - center)
    c = np.dot(point_on_line - center, point_on_line - center) - radius**2
    
    # Calculate the discriminant
    discriminant = b**2 - 4*a*c
    
    # Check if the line intersects the sphere
    if discriminant < 0:
        return 0, 0, np.array([]), np.array([]) # No intersection
    elif discriminant == 0:
        # One intersection point
        t = -b / (2*a)
        return t, 0, point_on_line + t * line_direction, np.array([])
    else:
        # Two intersection points
        t1 = (-b + np.sqrt(discriminant)) / (2*a)
        t2 = (-b - np.sqrt(discriminant)) / (2*a)
        return t1, t2, point_on_line + t1 * line_direction, point_on_line + t2 * line_direction

def varphi_x(x, x_bar):
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
    y = round(0.5*(np.linalg.norm(x-x_bar)**2), 4)
    return y

def choose_pt_giving_smallest_obj_value(x, x_bar, pt1, pt2):
    '''
        This function computes the objective value based on two critical points and picks the one
        that returns the smallest objective. The critical points are found via computing the
        intersection between a line and a sphere.
    '''

    val1 = varphi_x(x, pt1)
    val2 = varphi_x(x, pt2)
    val_true = varphi_x(x, x_bar)

    if val1 <= val2 and val1==val_true:
        return pt1
    elif val2 <= val1 and val2==val_true:
        return pt2
    else:
        return 0, print('SOMETHING IS INCORRECT. INTERSECTION VALUE IS NOT EQUAL TO THE TRUE VALUE.')
