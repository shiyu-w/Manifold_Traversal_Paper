import numpy as np
from abc import ABC, abstractmethod
import sys
import os
from PIL import Image
from sklearn.decomposition import PCA
import math
from utils.make_visualizations import return_only_P_final_swiss_roll, return_only_P_final_mobius, return_only_P_final_sphere_and_torus
from sklearn.datasets import fetch_openml
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt

sys.path.append('/Users/mariam/Documents/manifold_traversal')

def create_manifold(manifold_type, **kwargs):
    manifolds = {
        'sphere': Sphere,
        'torus': Torus,
        'swiss roll': SwissRoll,
        'mobius strip': MobiusStrip,
        'torus+island': TorusIsland,
        'gw': GW,
        'ALLMNIST': ALLMNIST, # all MNIST digits
        'MNISTOneDigit': MNISTOneDigit # one MNIST digit
    }
    
    if manifold_type not in manifolds:
        raise ValueError(f"Unknown manifold type: {manifold_type}")
    
    return manifolds[manifold_type](**kwargs)


class Manifold(ABC):
    @abstractmethod
    def generate_clean_samples(self, N):
        pass

    @abstractmethod
    def oracle_denoiser(self, x):
        pass

    @abstractmethod
    def plot_ground_truth_manifold(self, ax):
        pass

    @abstractmethod
    def visualization_embedding(self, q):
        pass

    @abstractmethod
    def center_axes(self, ax):
        pass

class Sphere(Manifold):
    def __init__(self, sphere_params=None):

        # Default parameters for the denoiser
        self.sphere_params = {
            'radius': 1} if sphere_params is None else sphere_params
        self.radius = self.sphere_params['radius']

    def generate_clean_samples(self, N):
        D = 3
        X_natural = np.random.randn(D, N)
        X_natural /= np.linalg.norm(X_natural, axis=0)
        X_natural *= self.radius
        return X_natural, D

    def oracle_denoiser(self, x):
        norm_x = np.linalg.norm(x)
        if norm_x > 1e-10:
            x_hat = x / norm_x
        else:
            x_hat = x
        return x_hat

    def plot_ground_truth_manifold(self, ax):
        numpts = 50
        theta = np.linspace(0, 2*np.pi, numpts)
        phi = np.linspace(-np.pi/2, np.pi/2, numpts)[..., np.newaxis]

        XX = np.cos(theta) * np.cos(phi)
        YY = np.sin(theta) * np.cos(phi)
        ZZ = np.ones((numpts, 1)) * np.sin(phi)
        ax.plot_surface(XX, YY, ZZ, color='purple', alpha=0.2)

    def visualization_embedding(self, q):
        q_vis = q
        P_vis = np.eye(3) 
        return q_vis, P_vis

    def center_axes(self, ax):
        ax.set_xlim([-1.05, 1.05])
        ax.set_ylim([-1.05, 1.05])
        ax.set_zlim([-1.05, 1.05])

class Torus(Manifold):
    def __init__(self, torus_params):
        self.torus_params = {
            'inner_radius': 5.0,
            'outer_radius': 2
            } if torus_params is None else torus_params
        
        self.inner_radius = self.torus_params['inner_radius']
        self.outer_radius = self.torus_params['outer_radius']

    def generate_clean_samples(self, N):
        theta = np.random.uniform(0.0, 2*np.pi, (1, N))
        phi = np.random.uniform(-np.pi, np.pi, (1, N))
        X_natural = np.zeros((3, N))
        X_natural[0, :] = np.cos(theta) * (self.inner_radius + self.outer_radius * np.cos(phi))
        X_natural[1, :] = np.sin(theta) * (self.inner_radius + self.outer_radius * np.cos(phi))
        X_natural[2, :] = self.outer_radius * np.sin(phi)
        D = 3

        return X_natural, D
    

    def plot_ground_truth_manifold(self, ax):
        numpts = 50
        theta = np.linspace(0, 2*np.pi, numpts)
        phi = np.linspace(-np.pi, np.pi, numpts)[..., np.newaxis]
        XX = np.cos(theta) * (self.inner_radius + self.outer_radius * np.cos(phi))
        YY = np.sin(theta) * (self.inner_radius + self.outer_radius * np.cos(phi))
        ZZ = self.outer_radius * np.ones((numpts, 1)) * np.sin(phi)
        ax.plot_surface(XX, YY, ZZ, color='purple', alpha=0.2)

    def visualization_embedding(self, q):
        q_vis = q
        P_vis = np.eye(3)
        return q_vis, P_vis

    def center_axes(self, ax):
        s = self.inner_radius + self.outer_radius
        ax.set_xlim([-s, s])
        ax.set_ylim([-s, s])
        ax.set_zlim([-s, s])

    def oracle_denoiser(self, x):
        theta = np.arctan2(x[1], x[0])
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)
        z = x.copy()
        z[0] = x[0] - cos_theta * self.inner_radius
        z[1] = x[1] - sin_theta * self.inner_radius

        if ( np.abs(cos_theta) >= np.abs(sin_theta) ) :
            phi = np.arctan2( z[2], z[0] / cos_theta )
        else:
            phi = np.arctan2( z[2], z[1] / sin_theta )

        x_hat = np.zeros( (3,))
        cos_phi, sin_phi = np.cos(phi), np.sin(phi)

        x_hat[0] = cos_theta * (self.inner_radius + self.outer_radius * cos_phi)
        x_hat[1] = sin_theta * (self.inner_radius + self.outer_radius * cos_phi)
        x_hat[2] = self.outer_radius * sin_phi

        return x_hat



class SwissRoll(Manifold):
    def __init__(self, max_height=1, theta_max=4*np.pi, swiss_roll_params=None):

        self.swiss_roll_params = {
            'max_height': 5.0,
            'theta_max': 14.5,
            'denoiser_params': {
                'learning_rate': 0.01,
                'max_iterations': 1000,
                'tolerance': 1e-6
            }
        } if swiss_roll_params is None else swiss_roll_params

        self.max_height = swiss_roll_params['max_height']
        self.theta_max = swiss_roll_params['theta_max']
        self.shift = 7
        self.denoiser_params = self.swiss_roll_params['denoiser_params']

    def generate_clean_samples(self, N):
        theta = np.random.uniform(0, self.theta_max, (1, N))
        v = self.max_height * np.random.uniform(-1.0, 1.0, size=(1, N))
        
        X_natural = np.zeros((3, N))
        X_natural[0, :] = theta * np.cos(theta)
        X_natural[1, :] = theta * np.sin(theta)
        X_natural[2, :] = v

        D = 3
        
        return X_natural, D
    

    def plot_ground_truth_manifold(self, ax):
        numpts = 500
        theta = np.linspace(0, self.theta_max, numpts)
        v = self.max_height * np.linspace(-1, 1, numpts)[..., np.newaxis]
        
        theta, v = np.meshgrid(theta, v)
        
        XX = theta * np.cos(theta)
        YY = theta * np.sin(theta)
        ZZ = v
        
        ax.plot_surface(XX, YY, ZZ, color='purple', alpha=0.2)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Swiss Roll Surface')


    def oracle_denoiser(self, x):
        params = {
            'learning_rate': self.denoiser_params['learning_rate'],
            'max_iterations': self.denoiser_params['max_iterations'],
            'tolerance': self.denoiser_params['tolerance'],
            'theta_max': self.theta_max,
            'v_max': self.max_height,
            'v_min': -self.max_height
        }

        x_bar = return_only_P_final_swiss_roll(x, **params)
        return x_bar

    def visualization_embedding(self, q):
        q_vis = q
        P_vis = np.eye(3)
        return q_vis, P_vis

    def center_axes(self, ax):
        ax.set_xlim([-self.shift - self.max_height, self.shift + self.max_height])
        ax.set_ylim([-self.shift - self.max_height, self.shift + self.max_height])
        ax.set_zlim([-self.shift - self.max_height, self.shift + self.max_height])



class MobiusStrip(Manifold):
    def __init__(self, R=1, mobius_params=None):

        # Default parameters for the denoiser
        self.mobius_params = {
                        'R': 1,
                        'limi': 1,
                        'dec': 3,
                        'num_samples': 100,
                        'num_local_mins': 50,
                        'theta_offset': 4e-7,
                        's_offset': 1e-6,
                        'learning_rate': 0.01,
                        'max_iterations': 1000,
                        'tolerance': 1e-6
                    } if mobius_params is None else mobius_params
        self.R = self.mobius_params['R']
        self.limi = self.mobius_params['limi']  # For center_axes

    def generate_clean_samples(self, N):
        s = np.random.uniform(-1, 1, N)
        t = np.random.uniform(0, 2 * np.pi, N)
        
        X_natural = np.zeros((3, N))
        X_natural[0, :] = (self.R + (s/2) * np.cos(t/2)) * np.cos(t)
        X_natural[1, :] = (self.R + (s/2) * np.cos(t/2)) * np.sin(t)
        X_natural[2, :] = (s/2) * np.sin(t/2)
        D = 3
        
        return X_natural, D
    


    def plot_ground_truth_manifold(self, ax):
        numpts = 500
        s = np.linspace(-1, 1, numpts)
        t = np.linspace(0, 2*np.pi, numpts)
        
        s, t = np.meshgrid(s, t)
        
        XX = (self.R + (s/2) * np.cos(t/2)) * np.cos(t)
        YY = (self.R + (s/2) * np.cos(t/2)) * np.sin(t)
        ZZ = (s/2) * np.sin(t/2)
        
        ax.plot_surface(XX, YY, ZZ, color='purple', alpha=0.2)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Mobius Strip Surface')

    def oracle_denoiser(self, x):
        params = {
            'dec': self.mobius_params['dec'],
            'num_samples': self.mobius_params['num_samples'],
            'num_local_mins': self.mobius_params['num_local_mins'],
            'theta_offset': self.mobius_params['theta_offset'],
            's_offset': self.mobius_params['s_offset'],
            'learning_rate': self.mobius_params['learning_rate'],
            'max_iterations': self.mobius_params['max_iterations'],
            'tolerance': self.mobius_params['tolerance'],
            'R': self.R
        }
        x_bar = return_only_P_final_mobius(x, **params)

        return x_bar
    


    def visualization_embedding(self, q):
        q_vis = q
        P_vis = np.eye(3)

        return q_vis, P_vis


    def center_axes(self, ax):
        ax.set_xlim([-self.limi, self.limi])
        ax.set_ylim([-self.limi, self.limi])
        ax.set_zlim([-self.limi, self.limi])



class TorusIsland(Manifold):
    def __init__(self, torus_params=None, sphere_params=None):
        # Default parameters for torus
        self.torus_params = {
            'inner_radius': 5.0,
            'outer_radius': 1.0
        } if torus_params is None else torus_params
        
        # Default parameters for sphere
        self.sphere_params = {
            'radius': 1.0
        } if sphere_params is None else sphere_params
        
        self.limi = 5  # For center_axes

    def generate_clean_samples(self, N):
        # Generate sphere samples
        r_sphere = self.sphere_params['radius']
        theta_sph = np.random.uniform(0, 2*np.pi, (1, N))
        phi_sph = np.random.uniform(0, 2*np.pi, (1, N))
        
        X_natural_sphere = np.zeros((3, N))
        X_natural_sphere[0, :] = r_sphere * np.cos(theta_sph) * np.sin(phi_sph)
        X_natural_sphere[1, :] = r_sphere * np.sin(theta_sph) * np.sin(phi_sph)
        X_natural_sphere[2, :] = r_sphere * np.cos(phi_sph)
        
        # Generate torus samples
        inner_radius = self.torus_params['inner_radius']
        outer_radius = self.torus_params['outer_radius']
        
        theta_torus = np.random.uniform(0.0, 2*np.pi, (1, N))
        phi_torus = np.random.uniform(-np.pi, np.pi, (1, N))
        
        X_natural_torus = np.zeros((3, N))
        X_natural_torus[0, :] = np.cos(theta_torus) * (inner_radius + outer_radius * np.cos(phi_torus))
        X_natural_torus[1, :] = np.sin(theta_torus) * (inner_radius + outer_radius * np.cos(phi_torus))
        X_natural_torus[2, :] = outer_radius * np.sin(phi_torus)
        
        # Combine and shuffle
        X_natural = np.concatenate((X_natural_sphere, X_natural_torus), axis=1)
        n_columns = X_natural.shape[1]
        shuffled_indices = np.random.permutation(n_columns)
        X_natural_shuffled = X_natural[:, shuffled_indices]
        D = 3
        
        return X_natural_shuffled, D




    def plot_ground_truth_manifold(self, ax):
        numpts = 50
        
        # Generate torus surface
        inner_radius = self.torus_params['inner_radius']
        outer_radius = self.torus_params['outer_radius']
        
        theta_torus = np.linspace(0, 2*np.pi, numpts)
        phi_torus = np.linspace(-np.pi, np.pi, numpts)
        theta_torus, phi_torus = np.meshgrid(theta_torus, phi_torus)
        
        XX_torus = np.cos(theta_torus) * (inner_radius + outer_radius * np.cos(phi_torus))
        YY_torus = np.sin(theta_torus) * (inner_radius + outer_radius * np.cos(phi_torus))
        ZZ_torus = outer_radius * np.ones((numpts, 1)) * np.sin(phi_torus)
        
        # Generate sphere surface
        theta_sph = np.linspace(0, 2*np.pi, numpts)
        phi_sph = np.linspace(-np.pi/2, np.pi/2, numpts)
        theta_sph, phi_sph = np.meshgrid(theta_sph, phi_sph)
        
        XX_sph = np.cos(theta_sph) * np.cos(phi_sph)
        YY_sph = np.sin(theta_sph) * np.cos(phi_sph)
        ZZ_sph = np.ones((numpts, 1)) * np.sin(phi_sph)
        
        # Combine surfaces
        XX = np.concatenate((XX_torus, XX_sph), axis=1)
        YY = np.concatenate((YY_torus, YY_sph), axis=1)
        ZZ = np.concatenate((ZZ_torus, ZZ_sph), axis=1)
        # print('XX = ', XX.shape)
        
        ax.plot_surface(XX, YY, ZZ, color='green', alpha=0.2)


    def oracle_denoiser(self, x):
        x_bar = return_only_P_final_sphere_and_torus(
            P=x, 
            R_torus=self.torus_params['inner_radius'],
            r_torus=self.torus_params['outer_radius'],
            r_sphere=self.sphere_params['radius']
        )
        return x_bar

    def visualization_embedding(self, q):
        q_vis = q
        P_vis = np.eye(3)

        return q_vis, P_vis


    def center_axes(self, ax):
        ax.set_xlim([-self.limi, self.limi])
        ax.set_ylim([-self.limi, self.limi])
        ax.set_zlim([-self.limi, self.limi])








class GW(Manifold):
    def __init__(self, gw_params = None):
        self.data = None
        self.pca_result = None
        self.centering_mean = None
        self.shift = 0.7 # PCA normalizes the data, [-1,1]
        self.gw_params = {
            'PCA_U': np.eye(2), # U matrix we get from pca
            'data_path': '/home/mariam/Desktop/gr_waves_mt/data' # data_path,
        } if gw_params is None else gw_params

    def generate_clean_samples(self, N):
        """Load wave data from numpy file"""
        temp_data = np.load(self.gw_params['data_path'])
        self.data = temp_data.T
        self._compute_pca(temp_data)
        D = 2048
        return self.data[:, :N], D


    def _compute_pca(self, data, n_components=3):
        """
        Internal method to compute PCA on the input data.
        
        Parameters:
        -----------
        data : numpy.ndarray
            Input data to perform PCA on
        n_components : int, optional
            Number of components (default: 3)
        """
        pca = PCA(n_components=n_components)
        self.pca_result = pca.fit_transform(data)
        self.centering_mean = pca.mean_


    def plot_ground_truth_manifold(self, ax, azim, elev, alpha):
        """
        Plot the PCA visualization of gravitational waves.
        
        Parameters:
        -----------
        ax : matplotlib.axes.Axes
            The 3D axes to plot on
        elev : int, optional
            Elevation viewing angle (default: 30)
        azim : int, optional
            Azimuth viewing angle (default: 45)
        """

        center_shift = self.gw_params['PCA_U'].T @ self.centering_mean

        ax.scatter(self.pca_result[:, 0] + center_shift[0],
                   self.pca_result[:, 1] + center_shift[1],
                   self.pca_result[:, 2] + center_shift[2],
                   c='purple',  # Color points by index
                   alpha=alpha)
        ax.view_init(elev=elev, azim=azim)
        
        # Set labels and title
        ax.set_xlabel('PCA feature 1')
        ax.set_ylabel('PCA feature 2')
        ax.set_zlabel('PCA feature 3')
        ax.set_title('PCA visualization of gravitational waves')

    def visualization_embedding(self, q):
        P_vis = self.gw_params['PCA_U']
        q_vis = np.dot(P_vis.T, q)
        return q_vis, P_vis

    def center_axes(self, ax):
        s = self.shift
        ax.set_xlim([-s, s])
        ax.set_ylim([-s, s])
        ax.set_zlim([-s, s])

    def oracle_denoiser(self, x, data):

        '''
            Returns the closest point to x on the manifold.
            Since we don't have the true manifold, we approximate it with a lot of clean samples and
            run exhaustive search over all clean samples points to find the nearest point
        '''

        # Find the closest point
        distances = np.sum((data.T - x).T**2, axis=0) # have to play around with dimensions using .T
        best_idx = np.argmin(distances)
        x_hat = data[:, best_idx]

        return x_hat




class ALLMNIST(Manifold):
    def __init__(self, mnist_params = None):
        self.data = None
        self.labels = None
        self.pca_result = None
        self.centering_mean = None
        self.shift = 1 # PCA normalizes the data, [-1,1]
        self.mnist_params = {} if mnist_params is None else mnist_params

    def generate_clean_samples(self, N):
        # Load the data via sklearn
        X_raw, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False) # shape is (70000, 784)
        X_tr = X_raw.T # shape is (784, 70000)

        # # SHUFFLE
        # # pick the first N columns after shuffling
        # n_columns = X_tr.shape[1]
        # shuffled_indices = np.random.permutation(n_columns)
        # X_shuffled = X_tr[:, shuffled_indices]
        # # Normalize the columns
        # self.data = X_shuffled[:, :N] / np.linalg.norm(X_shuffled[:, :N], axis=0, keepdims=True)

        # DONT SHUFFLE
        X_normalized = X_tr / np.linalg.norm(X_tr, axis=0, keepdims=True)
        self.data    = X_normalized[:, :N] 
        self.labels  = np.array(y, dtype='int')

        self._compute_pca(X_normalized.T)
        D = X_tr.shape[0]

        return (self.data, X_tr, self.labels, self.pca_result.T, self.mnist_params['PCA_U'],self.centering_mean, D)


    def _compute_pca(self, data, n_components=3):
        """
        Internal method to compute PCA on the input data.
        
        Parameters:
        -----------
        data : numpy.ndarray
            Input data to perform PCA on
        n_components : int, optional
            Number of components (default: 3)
        """
        pca = PCA(n_components=n_components)
        self.pca_result = pca.fit_transform(data)
        print('self.pca_result = ', self.pca_result.shape)

        self.mnist_params['PCA_U'] = pca.components_.T
        print('self.mnist_params[PCA_U] = ', self.mnist_params['PCA_U'].shape)
        print(self.mnist_params['PCA_U'])
        
        self.centering_mean = pca.mean_
        print('self.centering_mean = ', self.centering_mean.shape)


    def plot_ground_truth_manifold(self, ax, azim, elev, alpha):
        """
        Plot the PCA visualization of gravitational waves.
        
        Parameters:
        -----------
        ax : matplotlib.axes.Axes
            The 3D axes to plot on
        elev : int, optional
            Elevation viewing angle (default: 30)
        azim : int, optional
            Azimuth viewing angle (default: 45)
        """

        y_arr = np.array(self.labels, dtype=int)

        center_shift = self.mnist_params['PCA_U'].T @ self.centering_mean

        # fig = plt.figure(figsize=(8, 16))
        # ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.pca_result[:, 0] + center_shift[0],
                   self.pca_result[:, 1] + center_shift[1],
                   self.pca_result[:, 2] + center_shift[2],
                   c = y_arr,
                   cmap = 'tab10',
                   alpha = alpha)

        # Add legend
        legend_elements = [Line2D([0], [0], marker='o', color='w',
                                markerfacecolor=plt.cm.tab10(i / 10),
                                label=f'Digit {i}', markersize=10)
                        for i in range(10)]
        ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.1, 0.5))

        ax.view_init(elev=elev, azim=azim)
        # Set labels and title
        ax.set_xlabel('PCA feature 1')
        ax.set_ylabel('PCA feature 2')
        ax.set_zlabel('PCA feature 3')
        ax.set_title('PCA visualization of ALLMNIST Digits')
  


    def visualization_embedding(self, q):
        P_vis = self.mnist_params['PCA_U']
        q_vis = np.dot(P_vis.T, q)
        return q_vis, P_vis

    def center_axes(self, ax):
        s = self.shift
        ax.set_xlim([-s, s])
        ax.set_ylim([-s, s])
        ax.set_zlim([-s, s])

    def oracle_denoiser(self, x, data):

        '''
            Returns the closest point to x on the manifold.
            Since we don't have the true manifold, we approximate it with a lot of clean samples and
            run exhaustive search over all clean samples points to find the nearest point.

            Inputs:
                x ------------- noisy point of size (D,)
                data ---------- x_natural dataset of size (D, N)
        '''

        # Find the closest point
        distances = np.sum((data - x).T**2, axis=0) # have to play around with dimensions using .T
        best_idx = np.argmin(distances)
        x_hat = data[:, best_idx]

        return x_hat



class MNISTOneDigit(Manifold):
    def __init__(self, mnist_params = None):
        self.data = None
        self.pca_result = None
        self.centering_mean = None
        self.shift = 1 # PCA normalizes the data, [-1,1]
        self.mnist_params = {
            'PCA_U': None, # U matrix we get from pca,
            'digit': 5,
        } if mnist_params is None else mnist_params

    def generate_clean_samples(self, N):
        # Load the data via sklearn
        X_raw, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False) # shape is (70000, 784)
        X_tr = X_raw.T # shape is (784, 70000)
    
        # Get indices where y equals our desired digit
        digit = self.mnist_params['digit']
        digit_indices = (y == str(digit))  # y is stored as strings in MNIST

        # Select only those images
        X_digit = X_tr[:, digit_indices]  # shape (784, 6000)

        print(f"Number of images of digit {digit}: {X_digit.shape}")

        X_normalized = X_digit / np.linalg.norm(X_digit, axis=0, keepdims=True)
        self.data = X_normalized[:, :N] 
        self._compute_pca(X_normalized.T)
        D = X_digit.shape[0]

        return self.data, X_digit, y, D


    def _compute_pca(self, data, n_components=3):
        """
        Internal method to compute PCA on the input data.
        
        Parameters:
        -----------
        data : numpy.ndarray
            Input data to perform PCA on
        n_components : int, optional
            Number of components (default: 3)
        """
        pca = PCA(n_components=n_components)
        self.pca_result = pca.fit_transform(data)
        print('self.pca_result = ', self.pca_result.shape)

        self.mnist_params['PCA_U'] = pca.components_.T
        print('self.mnist_params[PCA_U]', self.mnist_params['PCA_U'].shape)
        
        self.centering_mean = pca.mean_
        print('self.centering_mean = ', self.centering_mean.shape)


    def plot_ground_truth_manifold(self, ax, azim, elev, alpha):
        """
        Plot the PCA visualization of gravitational waves.
        
        Parameters:
        -----------
        ax : matplotlib.axes.Axes
            The 3D axes to plot on
        elev : int, optional
            Elevation viewing angle (default: 30)
        azim : int, optional
            Azimuth viewing angle (default: 45)
        """

        center_shift = self.mnist_params['PCA_U'].T @ self.centering_mean
        print('PLOTTING GROUND TRUTH MANIFOLD')
        print('self.mnist_params[PCA_U]', self.mnist_params['PCA_U'].shape)

        ax.scatter(self.pca_result[:, 0] + center_shift[0],
                   self.pca_result[:, 1] + center_shift[1],
                   self.pca_result[:, 2] + center_shift[2],
                   c='purple', alpha=alpha)
        ax.view_init(elev=elev, azim=azim)
        
        # Set labels and title
        ax.set_xlabel('PCA feature 1')
        ax.set_ylabel('PCA feature 2')
        ax.set_zlabel('PCA feature 3')
        ax.set_title('PCA visualization of ALLMNIST Digits')

    def visualization_embedding(self, q):
        P_vis = self.mnist_params['PCA_U']
        q_vis = np.dot(P_vis.T, q)
        return q_vis, P_vis

    def center_axes(self, ax):
        s = self.shift
        ax.set_xlim([-s, s])
        ax.set_ylim([-s, s])
        ax.set_zlim([-s, s])

    def oracle_denoiser(self, x, data):

        '''
            Returns the closest point to x on the manifold.
            Since we don't have the true manifold, we approximate it with a lot of clean samples and
            run exhaustive search over all clean samples points to find the nearest point.

            Inputs:
                x ------------- noisy point of size (D,)
                data ---------- x_natural dataset of size (D, N)
        '''

        # Find the closest point
        distances = np.sum((data - x).T**2, axis=0) # have to play around with dimensions using .T
        best_idx = np.argmin(distances)
        x_hat = data[:, best_idx]

        return x_hat





