import numpy as np
import plotly.graph_objects as go

# Generate data for the torus
def torus_parametrization(R = 5, r = 2, n=100):
    theta = np.linspace(0, 2*np.pi, n)
    phi = np.linspace(0, 2*np.pi, n)
    theta, phi = np.meshgrid(theta, phi)
    x = (R + r*np.cos(phi)) * np.cos(theta)
    y = (R + r*np.cos(phi)) * np.sin(theta)
    z = r * np.sin(phi)
    return x, y, z

def find_angles_torus(P, R = 5):
    '''
        Given a point P in 3D, find angles theta and phi that
        are used to parametrize the torus.

        Returns angles in radians.
    '''
    # Unpack the given point
    x_P, y_P, z_P = P

    # Calculate theta (azimuthal angle in x-y plane from positive x-axis)
    # arctan2 chooses the quadrant correctly, is only defined for reals
    theta = np.arctan2(y_P, x_P)
    phi = np.arctan2(z_P, (np.sqrt(x_P**2 + y_P**2) - R))

    # Ensure theta and phi are in [0, 2π)
    theta = theta % (2 * np.pi)
    phi = phi % (2 * np.pi)

    return theta, phi

def find_x_bar_torus(P, R=5, r=2):
    '''
        This function takes a point P -- a numpy array of shape [x, y, z] coordinates,
        and returns its projection onto the torus characterized by radii R and r.
    '''
    # theta, phi = find_angles(P)
    theta, phi = find_angles_torus(P)

    # Compute the coordinates of the projection of P onto the torus
    x = (R + r*np.cos(phi))*np.cos(theta)
    y = (R + r*np.cos(phi))*np.sin(theta)
    z = r*np.sin(phi)

    P_bar = np.array([x, y, z])

    return P_bar


def generate_tangent_space_to_torus(P_bar, theta, phi):

    # Define the basis vectors
    u1 = np.array([-np.sin(theta), np.cos(theta), 0])
    u2 = np.array([-np.sin(phi)*np.cos(theta), -np.sin(phi)*np.sin(theta), np.cos(phi)])

    # Generate a grid of points
    grid_size = 10
    s = np.linspace(-3, 3, grid_size)
    t = np.linspace(-3, 3, grid_size)
    s, t = np.meshgrid(s, t)

    # Transform the grid points to lie on the plane
    x = s * u1[0] + t * u2[0] + P_bar[0]
    y = s * u1[1] + t * u2[1] + P_bar[1]
    z = s * u1[2] + t * u2[2] + P_bar[2]

    return x, y, z

def generate_normal_space_to_torus(point, theta, phi, t_range = np.linspace(-2,2,100)):

    # Define the basis vectors of the tangent space
    u_1 = np.array([-np.sin(theta), np.cos(theta), 0])
    u_2 = np.array([-np.sin(phi)*np.cos(theta), -np.sin(phi)*np.sin(theta), np.cos(phi)])

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

