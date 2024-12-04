import torch
import json
from models.mesh_xl.tokenizer import MeshTokenizer

# Dummy data
# Import the dataset class
from torch.utils.data import DataLoader
from datasets.dummy_dataset import Dataset


# Initialize dummy arguments
class Args:
    n_discrete_size = 128  # Discretization bins
    pad_id = -1  # Padding ID for invalid faces
    n_max_triangles = 10  # Maximum number of triangles


args = Args()
tokenizer = MeshTokenizer(args)

# Instantiate the dataset
dummy_dataset = Dataset()

# Create a DataLoader for batching
dataloader = DataLoader(dummy_dataset, batch_size=1, shuffle=True)

# Draw one input from the dataset
for data in dataloader:
    # Access the vertices and faces
    vertices = data["vertices"]
    faces = data["faces"]

    # Print the shapes of the drawn sample
    print("Vertices Shape:", vertices.shape)  # Should be (1, 100, 3) due to padding
    print("Faces Shape:", faces.shape)  # Should be (1, 100, 3) due to padding

    # Print the data itself
    print("Vertices:", vertices)
    print("Faces:", faces)

    # Break after one sample
    break


# Wrap data into a dictionary
data_dict = {"vertices": vertices, "faces": faces}

# Tokenize the dummy data
tokenized_data = tokenizer.tokenize(data_dict)


# Helper function to serialize a data dictionary
def serialize_data_dict(data_dict):
    """
    Converts PyTorch tensors in a dictionary to lists for JSON serialization.
    """
    return {
        key: value.tolist() if isinstance(value, torch.Tensor) else value
        for key, value in data_dict.items()
    }


# Combine input, tokenized, and reconstructed data into one dictionary
full_output = {
    "input_data": serialize_data_dict(data_dict),
    "tokenized_data": serialize_data_dict(tokenized_data),
}

# Save the combined data to a JSON file
output_file = "full_pipeline_output.json"
with open(output_file, "w") as f:
    json.dump(full_output, f, indent=4)

print(f"Full pipeline data saved to {output_file}")

# (Optional) Print the saved data for verification
with open(output_file, "r") as f:
    saved_data = json.load(f)

print(json.dumps(saved_data, indent=4))
