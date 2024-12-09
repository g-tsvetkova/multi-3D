import os
import numpy as np
import torch
from torch.utils.data import Dataset
import trimesh
import glob
from sklearn.model_selection import train_test_split


class Dataset:

    def __init__(self, *args, **kwargs):
        """
        Initialize the dataset with mesh files and preprocessing parameters.
        """
        super().__init__()
        self.max_vertices = 256
        self.max_faces = args.n_max_triangles
        self.all_files = glob.glob(
            "/root/.objaverse/filtered_meshes/*/*.glb"
        )  # Mesh file paths
        # Split the dataset into train/test
        self.train_files, self.val_files = train_test_split(
            self.mesh_files, test_size=0.2, random_state=42
        )

        # Set the split based on the argument passed (train or test)
        if args.split_set == "train":
            self.mesh_files = self.train_files
        elif args.split_set == "test":
            self.mesh_files = self.val_files
        else:
            raise ValueError("split_set must be 'train' or 'test'")

        print(f"Loading {args.split_set} data with {len(self.mesh_files)} files.")

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


if __name__ == "__main__":
    dataset = Dataset()
    sampler = torch.utils.data.RandomSampler(dataset)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=1,
        num_workers=4,
        # add for meshgpt
        drop_last=True,
    )
    next(iter(dataloader))
