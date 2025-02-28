import os
import numpy as np
from tqdm import tqdm
#from eval_utils.sample_generation import evaluate
from eval_utils.perplexity import evaluate
from datasets.base_dataset import BaseDataset, BASE_DIR

DATASET_DIR = os.path.join(BASE_DIR, 'MeshXL-shapenet-data')



class Dataset(BaseDataset):
    
    def __init__(self, args, split_set="train", augment=False): 
        super().__init__()
        
        # base dataset config
        self.dataset_name = 'shapenet_lamp'
        self.category_id = '03636649'
        self.eval_func = evaluate
        self.augment = augment and (split_set == 'train')
        self.num_repeat = 1
        self.pad_id = -1
        self.max_triangles = args.n_max_triangles
        self.max_vertices = self.max_triangles * 3
        
        # pre-load data into memory
        full_data = []
        for filename in tqdm(os.listdir(DATASET_DIR)):
            if self.category_id not in filename:
                continue
            if (split_set in filename) and filename.endswith('.npz'):
                loaded_data = np.load(
                    os.path.join(DATASET_DIR, filename),
                    allow_pickle=True
                )
                loaded_data = loaded_data["arr_0"].tolist()
                loaded_data = self._preprocess_data(loaded_data)
                full_data = full_data + loaded_data
        
        self.data = full_data
        
        print(f"[MeshDataset] Created from {len(self.data)} shapes for {self.dataset_name} {split_set}")


import torch
from models.mesh_xl.tokenizer import MeshTokenizer

if __name__ == "__main__":

    class Args:
        def __init__(self):
            self.n_discrete_size = 128
            self.n_max_triangles = 1000
            # plus any other needed args
    
    # Initialize dataset
    args = Args()
    dataset = Dataset(args, split_set="val")  # your dataset
    tokenizer = MeshTokenizer(args)  # or MeshTokenizer(args) if needed

    token_lengths = []
    for i, sample in enumerate(dataset.data):
        data_dict = {
            # Convert each NumPy array to a Torch Tensor, then unsqueeze(0)
            "vertices": torch.from_numpy(sample["vertices"]).unsqueeze(0),  # shape (1, nf, 3)
            "faces":    torch.from_numpy(sample["faces"]).unsqueeze(0),     # shape (1, nf, 3)
        }
        out_dict = tokenizer.tokenize(data_dict)
        length = out_dict["input_ids"].shape[1]
        token_lengths.append(length)

    print("Max token length across dataset:", max(token_lengths))
    print("Min token length across dataset:", min(token_lengths))
    print(f"Average token length: {sum(token_lengths)/len(token_lengths):.2f}")
