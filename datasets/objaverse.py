import os
import numpy as np
import torch
from torch.utils.data import Dataset
import trimesh
import glob


class Dataset:

    def __init__(self, *args, **kwargs):
        """
        Initialize the dataset with mesh files and preprocessing parameters.
        """
        super().__init__()
        self.max_vertices = 256
        self.max_faces = args.n_max_triangles
        self.all_files = glob.glob(
            f"{args.training_dir}/*/*/models/model_normalized.obj"
        )  # Mesh file paths

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        # Load the mesh
        mesh_file = self.all_files[idx]
        scene = trimesh.load(mesh_file, force="mesh")
        vertices, faces = scene.vertices, scene.faces

        # Pad vertices and faces**
        padded_vertices = np.zeros((self.max_vertices, 3), dtype=np.float32)
        padded_vertices[: vertices.shape[0]] = vertices

        padded_faces = np.zeros((self.max_faces, 3), dtype=np.int64)
        padded_faces[: faces.shape[0]] = faces

        # Construct data dictionary
        data_dict = {
            "vertices": torch.tensor(padded_vertices, dtype=torch.float32),
            "faces": torch.tensor(padded_faces, dtype=torch.long),
        }
        return data_dict

    @staticmethod
    def normalize_vertices(vertices):
        """
        Normalize vertices into a unit cube based on the mesh's longest axis.
        """
        min_coords = vertices.min(axis=0)
        max_coords = vertices.max(axis=0)
        center = (min_coords + max_coords) / 2
        longest_axis = (max_coords - min_coords).max()

        # Translate to center and scale to unit cube
        return (vertices - center) / longest_axis

    @staticmethod
    def reorder_vertices(vertices):
        """
        Re-order vertices by z, y, x coordinates.
        """
        # Sort vertices by z, y, x in ascending order
        sorted_indices = np.lexsort((vertices[:, 0], vertices[:, 1], vertices[:, 2]))
        return vertices[sorted_indices]
