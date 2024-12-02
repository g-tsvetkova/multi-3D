import trimesh

scene = trimesh.load(
    "/Users/zorin/Downloads/000074a334c541878360457c672b6c2e.glb", force="mesh"
)
# geometries = list(scene.geometry.values())
print(scene.vertices)
print(scene.faces)
