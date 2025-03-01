import os
import numpy as np
import torch
from torch.utils.data import Dataset
import trimesh
import glob
from sklearn.model_selection import train_test_split
from eval_utils.sample_generation import evaluate


def torch_lexical_sort(vertices: torch.Tensor) -> tuple:
    """
    Sort vertices by z, y, x coordinates lexicographically using PyTorch.

    Args:
        vertices: Tensor of shape (nv, 3)

    Returns:
        tuple: (sorted_vertices, sorting_indices)
            - sorted_vertices: Tensor of shape (nv, 3) with sorted vertices
            - sorting_indices: Tensor of shape (nv) tracking original indices
    """
    current_indices = torch.arange(vertices.shape[0], device=vertices.device)

    # Sort by x first, then y, then z
    x_sorted_indices = torch.argsort(vertices[:, 0], stable=True)
    vertices = vertices[x_sorted_indices]
    current_indices = current_indices[x_sorted_indices]

    y_sorted_indices = torch.argsort(vertices[:, 1], stable=True)
    vertices = vertices[y_sorted_indices]
    current_indices = current_indices[y_sorted_indices]

    z_sorted_indices = torch.argsort(vertices[:, 2], stable=True)
    sorted_vertices = vertices[z_sorted_indices]
    sorting_indices = current_indices[z_sorted_indices]

    return sorted_vertices, sorting_indices


# def reindex_faces_after_sort(
#     faces: torch.Tensor, sorting_indices: torch.Tensor
# ) -> torch.Tensor:
#     """
#     Reindex faces after sorting vertices to maintain correct triangle connections.

#     Args:
#         faces (torch.Tensor): Original faces tensor of shape (batch_size, nf, 3)
#         sorting_indices (torch.Tensor): Indices used to sort vertices of shape (batch_size, nv)

#     Returns:
#         torch.Tensor: Reindexed faces tensor with the same shape as input
#     """
#     # Validate input shapes
#     assert faces.shape[0] == sorting_indices.shape[0], "Batch sizes must match"

#     # Create an inverse mapping to track new indices
#     batch_size, num_vertices = sorting_indices.shape
#     reindexed_faces = torch.empty_like(faces)

#     # For each batch
#     for i in range(batch_size):
#         # Create inverse index mapping
#         # Create a tensor where index is the original vertex index and value is the new index
#         inverse_mapping = torch.empty(
#             num_vertices, dtype=torch.long, device=sorting_indices.device
#         )
#         inverse_mapping[sorting_indices[i]] = torch.arange(
#             num_vertices, device=sorting_indices.device
#         )
#         # Reindex the faces for this batch using the inverse mapping
#         reindexed_faces[i] = inverse_mapping[faces[i]]

#     return reindexed_faces

def reindex_faces_after_sort(faces: torch.Tensor, sorting_indices: torch.Tensor) -> torch.Tensor:
    """
    Reindex faces after sorting vertices to maintain correct triangle connections.

    Args:
        faces (torch.Tensor): Original faces tensor of shape (nf, 3)
        sorting_indices (torch.Tensor): Indices used to sort vertices of shape (nv)

    Returns:
        torch.Tensor: Reindexed faces tensor of shape (nf, 3)
    """
    # Create inverse index mapping
    inverse_mapping = torch.empty(sorting_indices.shape[0], dtype=torch.long, device=sorting_indices.device)
    inverse_mapping[sorting_indices] = torch.arange(sorting_indices.shape[0], device=sorting_indices.device)

    # Reindex faces
    reindexed_faces = inverse_mapping[faces]
    return reindexed_faces



# def normalize_vertices_scale(vertices: torch.Tensor) -> torch.Tensor:
#     """Scale vertices so that the long diagonal of the bounding box is one.

#     Args:
#         vertices: unscaled vertices of shape (batch_size, num_vertices, 3)
#     Returns:
#         scaled_vertices: scaled vertices of shape (batch_size, num_vertices, 3)
#     """
#     # Compute min and max per batch along the vertex dimension
#     vert_min, _ = torch.min(vertices, dim=1, keepdim=True)  # Shape: (batch_size, 1, 3)
#     vert_max, _ = torch.max(vertices, dim=1, keepdim=True)  # Shape: (batch_size, 1, 3)

#     # Compute extents and scale
#     extents = vert_max - vert_min  # Shape: (batch_size, 1, 3)
#     scale = torch.sqrt(
#         torch.sum(extents**2, dim=-1, keepdim=True)
#     )  # Shape: (batch_size, 1, 1)

#     # Normalize vertices
#     scaled_vertices = vertices / scale  # Broadcasting scales appropriately
#     return scaled_vertices

def normalize_vertices_scale(vertices: torch.Tensor) -> torch.Tensor:
    # vertices: (num_vertices, 3)
    # Compute min and max along the vertex dimension (dim=0)
    vert_min = torch.min(vertices, dim=0, keepdim=True)[0]  # (1, 3)
    vert_max = torch.max(vertices, dim=0, keepdim=True)[0]  # (1, 3)

    # Compute extents and scale factor
    extents = vert_max - vert_min  # (1, 3)
    scale = torch.sqrt(torch.sum(extents**2, dim=-1, keepdim=True))  # (1, 1)
    
    # Normalize
    scaled_vertices = vertices / scale  # Broadcasted division
    return scaled_vertices



class Dataset:

    def __init__(self, args, split_set="train", **kwargs):
        """
        Initialize the dataset with mesh files and preprocessing parameters.
        """
        super().__init__()
        # self.max_vertices = 256
        # self.max_faces = args.n_max_triangles
        self.mesh_files = glob.glob(
            "/root/.objaverse/filtered_meshes/*/*.glb"
        )  # Mesh file paths
        # Split the dataset into train/test
        self.train_files, self.val_files = train_test_split(
            self.mesh_files, test_size=0.2, random_state=42
        )

        # Set the split based on the argument passed (train or test)
        if split_set == "train":
            self.mesh_files = self.train_files
        elif split_set == "test":
            self.mesh_files = self.val_files
        else:
            raise ValueError("split_set must be 'train' or 'test'")

        print(f"Loading {split_set} data with {len(self.mesh_files)} files.")
        
        # Define maximum dimensions for padding (adjust these as needed)
        self.max_vertices = 2500
        self.max_faces = args.n_max_triangles
        self.face_pad_id = -1  # Use -1 to indicate padded faces

        self.eval_func = evaluate

    def __len__(self):
        return len(self.mesh_files)

    def __getitem__(self, idx):
        # Load the mesh
        mesh_file = self.mesh_files[idx]
        scene = trimesh.load(mesh_file, force="mesh")
        vertices, faces = scene.vertices, scene.faces

        # convert to torch tensors
        vertices = torch.from_numpy(vertices).float()
        faces = torch.from_numpy(faces).long()

        # Preprocessing: normalize, reorder vertices, and reorder faces
        vertices = normalize_vertices_scale(vertices)
        # print("Normalized vertices: ", vertices)

        # Reorder vertices by (z, y, x) using lexicographical sorting
        vertices, new_indices = torch_lexical_sort(vertices)
        # print("Sorted vertices: ", vertices)

        # Reindex faces after sorting vertices
        reindexed_faces = reindex_faces_after_sort(faces, new_indices)
        # print("Reindexed faces: ", reindexed_faces)
        # After sorting and reindexing:
        # Reorder faces by (z, y, x) using lexicographical sorting
        sorted_faces, _ = torch_lexical_sort(reindexed_faces)
        # print("Sorted faces: ", sorted_faces)

        # Pad vertices
        padded_vertices = torch.zeros((self.max_vertices, 3), dtype=vertices.dtype)
        num_v = min(vertices.shape[0], self.max_vertices)
        padded_vertices[:num_v, :] = vertices[:num_v]

        # Pad faces
        padded_faces = torch.full((self.max_faces, 3), fill_value=self.face_pad_id, dtype=faces.dtype)
        num_f = min(reindexed_faces.shape[0], self.max_faces)
        padded_faces[:num_f, :] = reindexed_faces[:num_f]

        data_dict = {
            "vertices": padded_vertices,   # (max_vertices, 3)
            "faces": padded_faces,         # (max_faces, 3), with padding
            "shape_idx": torch.tensor(idx, dtype=torch.int64),
            "num_vertices": torch.tensor(num_v, dtype=torch.int64),
            "num_faces": torch.tensor(num_f, dtype=torch.int64)
        }

        return data_dict

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
