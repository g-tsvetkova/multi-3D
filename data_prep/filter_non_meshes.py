import os
import trimesh
import shutil
from tqdm import tqdm

def is_polygonal_mesh(file_path):
    """
    Check if the file at file_path is a polygonal mesh with faces.
    """
    try:
        scene = trimesh.load(file_path, force='mesh')
        # Check if the loaded scene has faces
        # Some objects might load as Path3D which has no 'faces' attribute
        return hasattr(scene, 'faces') and len(scene.faces) > 0
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return False

def main():
    # Directory containing the filtered meshes
    source_dir = "/root/.objaverse/filtered_meshes/"
    
    # directory to move non-mesh files instead of deleting them
    non_mesh_dir = os.path.join(source_dir, "non_meshes")
    os.makedirs(non_mesh_dir, exist_ok=True)

    # Scan through the directory structure
    for root, dirs, files in os.walk(source_dir):
        # Skip the "non_meshes" directory
        if non_mesh_dir in root:
            continue

        for f in tqdm(files, desc=f"Checking in {root}"):
            file_path = os.path.join(root, f)

            # Check file extension first (optional)
            ext = os.path.splitext(f)[1].lower()
            if ext not in ['.glb', '.obj', '.stl', '.ply', '.gltf']:
                continue

            # Check if polygonal mesh
            if not is_polygonal_mesh(file_path):
                print(f"Non-mesh file detected: {file_path}")
                
                # Move to non_mesh_dir
                rel_path = os.path.relpath(root, source_dir)
                dest_subdir = os.path.join(non_mesh_dir, rel_path)
                os.makedirs(dest_subdir, exist_ok=True)
                dest_path = os.path.join(dest_subdir, f)
                
                shutil.move(file_path, dest_path)
                print(f"Moved {file_path} to {dest_path}")

if __name__ == "__main__":
    main()
