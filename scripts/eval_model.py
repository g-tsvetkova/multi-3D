import os
import torch
import numpy as np
from tqdm import tqdm
import trimesh
from models.mesh_xl.model import get_model
from datasets.base_dataset import Dataset
from eval_utils.evaluation_metrics import compute_all_metrics, jsd_between_point_cloud_sets

def sample_points_from_ply(ply_path, n_points=2048):
    """Sample points from generated mesh PLY files"""
    mesh = trimesh.load(ply_path)
    points, _ = trimesh.sample.sample_surface(mesh, n_points)
    return points.astype(np.float32)

def load_generated_samples(generated_dir, num_samples=64):
    """Load and process generated samples"""
    all_files = [f for f in os.listdir(generated_dir) if f.endswith('.ply')]
    generated_pcs = []
    
    for filename in tqdm(all_files[:num_samples], desc='Processing Generated'):
        pc = sample_points_from_ply(os.path.join(generated_dir, filename))
        generated_pcs.append(pc)
    
    return torch.tensor(np.array(generated_pcs))

def load_reference_data(dataset):
    """Load reference point clouds from validation set"""
    ref_pcs = []
    for sample in tqdm(dataset.data, desc='Loading Reference'):
        # Convert mesh to point cloud (replace with your actual preprocessing)
        vertices = sample['vertices'].numpy()
        faces = sample['faces'].numpy()
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        pc, _ = trimesh.sample.sample_surface(mesh, 2048)
        ref_pcs.append(pc.astype(np.float32))
    
    return torch.tensor(np.array(ref_pcs))

def main(args):
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generated_dir = os.path.join(args.checkpoint_dir, 'sampled')
    num_samples = 64  # Match validation set size

    # Load reference data
    val_dataset = Dataset(args, split_set="val")
    ref_pcs = load_reference_data(val_dataset).to(device)

    # Load generated samples
    gen_pcs = load_generated_samples(generated_dir, num_samples).to(device)

    # Compute metrics
    metrics = compute_all_metrics(gen_pcs, ref_pcs, batch_size=16)
    
    # Compute JSD
    metrics['JSD'] = jsd_between_point_cloud_sets(
        gen_pcs.cpu().numpy(),
        ref_pcs.cpu().numpy()
    )

    # Print results
    print("\nFinal Evaluation Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    class Args:
        checkpoint_dir = "./checkpoints"
        n_max_triangles = 1000
        # Add other required arguments here
    
    main(Args())