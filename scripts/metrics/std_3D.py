'''
Adapted from Achlioptas et. al (2018)
Learning representations and generative models for 3d point clouds. 
https://github.com/optas/latent_3d_points/blob/master/notebooks/compute_evaluation_metrics.ipynb
'''

import numpy as np
import torch
import trimesh
from tqdm import tqdm
# from sklearn.neighbors import NearestNeighbors
from scipy.stats import entropy
from eval_utils.eval_metrics import *


def sample_points(mesh_path, num_points = 2048):
    """
    Sample uniformly distributed points from a mesh surface.

    Returns samples ((count, 3) float)
    """
    mesh = trimesh.load(mesh_path)
    points = mesh.sample(num_points)
    return points


def chamfer_distance(p1, p2):
    """
    Compute the Chamfer distance between two point clouds.
    """
    

def coverage_cd(p1, p2):
    """
    Computes the coverage.

    The fraction of point clouds in set B that were matched to point clouds in
    set A.
    """
    p1 = p1.cpu().numpy()
    p2 = p2.cpu().numpy()

    cov_score = 0
    return cov_score * 10**-3

def mmd_cd(p1, p2):
    """
    Computes the Minimum Matching Distance (MMD) between two point clouds.
    """
    p1 = p1.cpu().numpy()
    p2 = p2.cpu().numpy()
    mmd_score = 0
    return mmd_score * 10**-3

# JSD adapted from https://github.com/optas/latent_3d_points
def jsd_between_point_cloud_sets(
        sample_pcs, ref_pcs, resolution=28):
    """Computes the JSD between two sets of point-clouds,
       as introduced in the paper
    ```Learning Representations And Generative Models For 3D Point Clouds```.
    Args:
        sample_pcs: (np.ndarray S1xR2x3) S1 point-clouds, each of R1 points.
        ref_pcs: (np.ndarray S2xR2x3) S2 point-clouds, each of R2 points.
        resolution: (int) grid-resolution. Affects granularity of measurements.
    """
    in_unit_sphere = True
    sample_grid_var = entropy_of_occupancy_grid(
        sample_pcs, resolution, in_unit_sphere)[1]
    ref_grid_var = entropy_of_occupancy_grid(
        ref_pcs, resolution, in_unit_sphere)[1]
    return jensen_shannon_divergence(sample_grid_var, ref_grid_var)


def entropy_of_occupancy_grid(
        pclouds, grid_resolution, in_sphere=False, verbose=False):
    """Given a collection of point-clouds, estimate the entropy of
    the random variables corresponding to occupancy-grid activation patterns.
    Inputs:
        pclouds: (numpy array) #point-clouds x points per point-cloud x 3
        grid_resolution (int) size of occupancy grid that will be used.
    """
    epsilon = 10e-4
    bound = 0.5 + epsilon
    if abs(np.max(pclouds)) > bound or abs(np.min(pclouds)) > bound:
        if verbose:
            warnings.warn('Point-clouds are not in unit cube.')

    if in_sphere and np.max(np.sqrt(np.sum(pclouds ** 2, axis=2))) > bound:
        if verbose:
            warnings.warn('Point-clouds are not in unit sphere.')

    grid_coordinates, _ = unit_cube_grid_point_cloud(grid_resolution, in_sphere)
    grid_coordinates = grid_coordinates.reshape(-1, 3)
    grid_counters = np.zeros(len(grid_coordinates))
    grid_bernoulli_rvars = np.zeros(len(grid_coordinates))
    nn = NearestNeighbors(n_neighbors=1).fit(grid_coordinates)

    for pc in tqdm(pclouds, desc='JSD'):
        _, indices = nn.kneighbors(pc)
        indices = np.squeeze(indices)
        for i in indices:
            grid_counters[i] += 1
        indices = np.unique(indices)
        for i in indices:
            grid_bernoulli_rvars[i] += 1

    acc_entropy = 0.0
    n = float(len(pclouds))
    for g in grid_bernoulli_rvars:
        if g > 0:
            p = float(g) / n
            acc_entropy += entropy([p, 1.0 - p])

    return acc_entropy / len(grid_counters), grid_counters


def jensen_shannon_divergence(P, Q):
    if np.any(P < 0) or np.any(Q < 0):
        raise ValueError('Negative values.')
    if len(P) != len(Q):
        raise ValueError('Non equal size.')

    P_ = P / np.sum(P)  # Ensure probabilities.
    Q_ = Q / np.sum(Q)

    e1 = entropy(P_, base=2)
    e2 = entropy(Q_, base=2)
    e_sum = entropy((P_ + Q_) / 2.0, base=2)
    res = e_sum - ((e1 + e2) / 2.0)

    res2 = _jsdiv(P_, Q_)

    if not np.allclose(res, res2, atol=10e-5, rtol=0):
        warnings.warn('Numerical values of two JSD methods don\'t agree.')

    return res

      
    
if __name__ == "__main__":

    generated_mesh_path = "data/generated_meshes/generated_mesh.ply"
    reference_mesh_path = "data/reference_meshes/reference_mesh.ply"

    # load generated meshes

    # load reference meshes

    # sample point clouds from generated meshes
    sample_pcs = sample_points(generated_mesh_path)
    print("Number of point clouds sampled from generated meshes: ", len(sample_pcs))

    # sample point clouds from reference meshes
    ref_pcs = sample_points(reference_mesh_path)
    print("Number of point clouds sampled from reference meshes: ", len(ref_pcs))

    # compute chamfer distance
    cd_score = chamfer_distance(sample_pcs, ref_pcs)
    print("Chamfer distance: ", cd_score)

    # compute coverage

    # compute mmd

    pass