import os
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from eval_utils.sample_generation import evaluate
import glob
import torch
from torch.utils.data import Dataset


class Dataset:

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.eval_func = evaluate
        # self.all_files = glob.glob(
        #     f"{self.training_dir}/*/*/models/model_normalized.obj"
        # )

    def __len__(self):
        return 10

    def __getitem__(self, idx):
        nv = np.random.randint(0, 10)  # Number of vertices
        nf = np.random.randint(0, 10)  # Number of faces

        # Generate random vertices and faces
        vertices = np.random.uniform(0, 1, size=(nv, 3)).astype(np.float32)
        faces = np.random.randint(0, nv, size=(nf, 3)).astype(np.int64)

        # Pad vertices to max_vertices
        padded_vertices = np.zeros(
            (10, 3), dtype=np.float32
        )  # Neutral padding with zeros
        padded_vertices[:nv] = vertices  # Fill actual vertices

        # Pad faces to max_faces
        padded_faces = np.zeros((10, 3), dtype=np.int64)  # Neutral padding with zeros
        padded_faces[:nf] = faces  # Fill actual faces

        # Construct data dictionary
        data_dict = {
            "vertices": torch.tensor(padded_vertices, dtype=torch.float32),
            "faces": torch.tensor(padded_faces, dtype=torch.long),
        }
        return data_dict
