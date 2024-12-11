"""
Filter out meshes with more than 800 faces
"""

import os
import trimesh
import shutil
from tqdm import tqdm
import multiprocessing
from functools import partial


def count_faces(file_path):
    """
    Count the number of faces in a mesh file.
    
    Args:
        file_path (str): Path to the mesh file
    
    Returns:
        int: Number of faces in the mesh, or -1 if there's an error
    """
    try:
        mesh = trimesh.load(file_path, force='mesh')
        return len(mesh.faces)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return -1

def filter_mesh_directory(
    source_dir, 
    output_dir, 
    max_faces=800, 
    verbose=True
):
    """
    Filter mesh files in a directory, preserving directory structure.
    
    Args:
        source_dir (str): Root directory containing mesh files
        output_dir (str): Destination directory for filtered files
        max_faces (int): Maximum number of faces allowed
        verbose (bool): Whether to print detailed processing information
    
    Returns:
        dict: Filtering statistics
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Tracking statistics
    stats = {
        'total_files': 0,
        'processed_files': 0,
        'filtered_out_files': 0,
        'filtered_files': 0,
        'face_count_details': {
            'min': float('inf'),
            'max': 0,
            'total': 0
        }
    }
    
    # Walk through the source directory
    for root, dirs, files in os.walk(source_dir):
        # Skip if no files in this directory
        if not files:
            continue
        
        # Corresponding output subdirectory
        relative_path = os.path.relpath(root, source_dir)
        output_subdir = os.path.join(output_dir, relative_path)
        
        # Create corresponding subdirectory in output
        os.makedirs(output_subdir, exist_ok=True)
        
        # Process files in this directory
        for filename in tqdm(files, desc=f"Processing {relative_path}"):
            # Skip non-mesh files
            if not filename.lower().endswith(('.glb', '.obj', '.stl')):
                continue
            
            # Full paths
            source_file = os.path.join(root, filename)
            output_file = os.path.join(output_subdir, filename)
            
            # Count faces
            face_count = count_faces(source_file)
            
            # Update statistics
            stats['total_files'] += 1
            stats['processed_files'] += 1
            
            # Update face count details
            stats['face_count_details']['min'] = min(
                stats['face_count_details']['min'], 
                face_count
            )
            stats['face_count_details']['max'] = max(
                stats['face_count_details']['max'], 
                face_count
            )
            stats['face_count_details']['total'] += face_count
            
            # Filter logic
            if verbose:
                print(f"File: {filename}, Faces: {face_count}")
            
            if face_count <= max_faces:
                # Keep the file in its original structure
                shutil.copy2(source_file, output_file)
                stats['filtered_files'] += 1
            else:
                # Filtered out
                stats['filtered_out_files'] += 1
    
    # Calculate average face count
    if stats['processed_files'] > 0:
        stats['average_face_count'] = (
            stats['face_count_details']['total'] / stats['processed_files']
        )
    
    return stats

def main():
    # Configurable parameters
    SOURCE_DIR = '/root/.objaverse/hf-objaverse-v1/glbs'
    OUTPUT_DIR = '/root/.objaverse/filtered_meshes'
    MAX_FACES = 800
    
    # Run filtering
    print("Starting mesh filtering process...")
    results = filter_mesh_directory(
        source_dir=SOURCE_DIR, 
        output_dir=OUTPUT_DIR, 
        max_faces=MAX_FACES,
        verbose=True
    )
    
    # Print detailed statistics
    print("\n--- Filtering Statistics ---")
    print(f"Total Files Processed: {results['total_files']}")
    print(f"Filtered Files Kept: {results['filtered_files']}")
    print(f"Filtered Out Files: {results['filtered_out_files']}")
    print("\nFace Count Details:")
    print(f"Minimum Faces: {results['face_count_details']['min']}")
    print(f"Maximum Faces: {results['face_count_details']['max']}")
    print(f"Average Faces: {results.get('average_face_count', 0):.2f}")

if __name__ == '__main__':
    main()