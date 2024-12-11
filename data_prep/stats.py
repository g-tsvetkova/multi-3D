import argparse
import os
import trimesh
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

def get_mesh_files(root_dir):
    """
    Recursively find all mesh files in the given directory.
    """
    mesh_files = []
    valid_extensions = {'.glb', '.obj', '.stl', '.ply', '.gltf'}
    for root, dirs, files in os.walk(root_dir):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in valid_extensions:
                mesh_files.append(os.path.join(root, f))
    return mesh_files

def load_mesh_info(file_path):
    """
    Load a mesh and return number of vertices and faces.
    Returns (num_vertices, num_faces) or (None, None) if error.
    """
    try:
        mesh = trimesh.load(file_path, force='mesh')
        print(mesh)
        return len(mesh.vertices), len(mesh.faces)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None

def print_distribution_stats(data, name="Data"):
    """
    Print statistical distribution details for given data array.
    """
    if len(data) == 0:
        print(f"No {name.lower()} data available.")
        return

    data = np.array(data)
    min_val = np.min(data)
    max_val = np.max(data)
    avg_val = np.mean(data)
    med_val = np.median(data)

    # Compute quantiles
    q25 = np.percentile(data, 25)
    q50 = np.percentile(data, 50)  # same as median
    q75 = np.percentile(data, 75)
    q90 = np.percentile(data, 90)

    print(f"\n--- {name} Distribution ---")
    print(f"Count: {len(data)}")
    print(f"Min: {min_val}")
    print(f"25th percentile: {q25}")
    print(f"Median (50th): {q50}")
    print(f"75th percentile: {q75}")
    print(f"90th percentile: {q90}")
    print(f"Max: {max_val}")
    print(f"Average: {avg_val:.2f}")

def main():
    parser = argparse.ArgumentParser(description="Compute and plot vertex/face distributions for meshes.")
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help='Path to the dataset directory containing mesh files.')
    parser.add_argument('--output_dir', type=str, default='.',
                        help='Directory to save the generated plots and results.')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Gather all mesh files
    print("Scanning for mesh files...")
    mesh_files = get_mesh_files(args.dataset_dir)
    print(f"Found {len(mesh_files)} mesh files.")

    vertex_counts = []
    face_counts = []

    # Process each mesh file
    for mf in tqdm(mesh_files, desc="Processing meshes"):
        nv, nf = load_mesh_info(mf)
        if nv is None or nf is None:
            continue  # skip invalid meshes

        vertex_counts.append(nv)
        face_counts.append(nf)

    # Print distribution statistics
    print_distribution_stats(vertex_counts, "Vertex Count")
    print_distribution_stats(face_counts, "Face Count")

    # Plot distributions if data is available
    if len(vertex_counts) > 0:
        plt.figure()
        plt.hist(vertex_counts, bins=50, color='blue', alpha=0.7)
        plt.title('Vertex Count Distribution')
        plt.xlabel('Number of Vertices')
        plt.ylabel('Frequency')
        plt.grid(True)
        vertex_plot_path = os.path.join(args.output_dir, 'vertex_distribution.png')
        plt.savefig(vertex_plot_path)
        plt.close()
        print(f"\nSaved vertex distribution plot to {vertex_plot_path}")

    if len(face_counts) > 0:
        plt.figure()
        plt.hist(face_counts, bins=50, color='green', alpha=0.7)
        plt.title('Face Count Distribution')
        plt.xlabel('Number of Faces')
        plt.ylabel('Frequency')
        plt.grid(True)
        face_plot_path = os.path.join(args.output_dir, 'face_distribution.png')
        plt.savefig(face_plot_path)
        plt.close()
        print(f"Saved face distribution plot to {face_plot_path}")

if __name__ == '__main__':
    main()
