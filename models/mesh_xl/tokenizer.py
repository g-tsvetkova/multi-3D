import torch
from torch import nn, Tensor
from typing import Tuple
from einops import rearrange, repeat, reduce


def discretize(
    t: Tensor,
    continuous_range: Tuple[float, float],
    num_discrete: int = 128,
) -> Tensor:
    lo, hi = continuous_range
    assert hi > lo
    t = (t - lo) / (hi - lo)  # Normalize to [0, 1]
    t *= num_discrete  # Scale to bins
    t -= 0.5  # Adjust for rounding
    return t.round().long().clamp(min=0, max=num_discrete - 1)


def undiscretize(
    t: Tensor,
    continuous_range: Tuple[float, float],
    num_discrete: int = 128,
) -> Tensor:
    lo, hi = continuous_range
    assert hi > lo
    t = t.float()
    t += 0.5
    t /= num_discrete
    return t * (hi - lo) + lo


def torch_lexical_sort(vertices: torch.Tensor) -> tuple:
    """
    Sort vertices by z, y, x coordinates lexicographically using PyTorch.

    Args:
        vertices: Tensor of shape (batch_size, nv, 3)

    Returns:
        tuple: (sorted_vertices, sorting_indices)
            - sorted_vertices: Tensor of shape (batch_size, nv, 3) with sorted vertices
            - sorting_indices: Tensor of shape (batch_size, nv) tracking original indices
    """
    # Create placeholder for sorted vertices and sorting indices
    sorted_vertices = torch.empty_like(vertices)
    sorting_indices = torch.empty(
        vertices.shape[:2], dtype=torch.long, device=vertices.device
    )

    for i, batch in enumerate(vertices):  # Process each batch independently
        # Track sorting indices for this batch
        current_indices = torch.arange(batch.shape[0], device=batch.device)

        # Sort by x first, then y, then z to achieve z > y > x lexicographical sort
        x_sorted_indices = torch.argsort(batch[:, 0], stable=True)
        batch = batch[x_sorted_indices]
        current_indices = current_indices[x_sorted_indices]

        y_sorted_indices = torch.argsort(batch[:, 1], stable=True)
        batch = batch[y_sorted_indices]
        current_indices = current_indices[y_sorted_indices]

        z_sorted_indices = torch.argsort(batch[:, 2], stable=True)
        sorted_vertices[i] = batch[z_sorted_indices]
        sorting_indices[i] = current_indices[z_sorted_indices]

    return sorted_vertices, sorting_indices


def reindex_faces_after_sort(
    faces: torch.Tensor, sorting_indices: torch.Tensor
) -> torch.Tensor:
    """
    Reindex faces after sorting vertices to maintain correct triangle connections.

    Args:
        faces (torch.Tensor): Original faces tensor of shape (batch_size, nf, 3)
        sorting_indices (torch.Tensor): Indices used to sort vertices of shape (batch_size, nv)

    Returns:
        torch.Tensor: Reindexed faces tensor with the same shape as input
    """
    # Validate input shapes
    assert faces.shape[0] == sorting_indices.shape[0], "Batch sizes must match"

    # Create an inverse mapping to track new indices
    batch_size, num_vertices = sorting_indices.shape
    reindexed_faces = torch.empty_like(faces)

    # For each batch
    for i in range(batch_size):
        # Create inverse index mapping
        # Create a tensor where index is the original vertex index and value is the new index
        inverse_mapping = torch.empty(
            num_vertices, dtype=torch.long, device=sorting_indices.device
        )
        inverse_mapping[sorting_indices[i]] = torch.arange(
            num_vertices, device=sorting_indices.device
        )
        # Reindex the faces for this batch using the inverse mapping
        reindexed_faces[i] = inverse_mapping[faces[i]]

    return reindexed_faces


class MeshTokenizer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.pad_id = -1
        self.num_discrete_coors = args.n_discrete_size  # default: 128
        self.codebook_size = args.n_discrete_size  # default: 128
        self.coor_continuous_range = (-1.0, 1.0)

    @staticmethod
    def normalize_vertices(vertices: Tensor) -> Tensor:
        """
        Normalize vertices into a unit cube based on the longest axis.
        """
        min_coords = vertices.min(dim=0).values
        max_coords = vertices.max(dim=0).values
        center = (min_coords + max_coords) / 2
        longest_axis = (max_coords - min_coords).max()
        return (vertices - center) / longest_axis

    def tokenize(self, data_dict: dict) -> dict:
        """
        Turn 3D meshes into sequential tokens: <bos> [<x>, <y>, <z>], ... <eos>.
        """
        # Extract vertices and faces
        vertices = data_dict["vertices"]  # shape: batch x nv x 3
        faces = data_dict["faces"]  # shape: batch x nf x 3
        print("Raw vertices: ", vertices)
        print("Raw faces: ", faces)

        # Preprocessing: normalize, reorder vertices, and reorder faces
        vertices = self.normalize_vertices(vertices)
        print("Normalized vertices: ", vertices)

        # Reorder vertices by (z, y, x) using lexicographical sorting
        vertices, new_indices = torch_lexical_sort(vertices)
        print("Sorted vertices: ", vertices)

        # Reindex faces after sorting vertices
        reindexed_faces = reindex_faces_after_sort(faces, new_indices)
        print("Reindexed faces: ", reindexed_faces)

        # Reorder faces by (z, y, x) using lexicographical sorting
        sorted_faces, _ = torch_lexical_sort(reindexed_faces)
        print("Sorted faces: ", sorted_faces)

        # Generate face mask
        face_mask = reduce(faces != self.pad_id, "b nf c -> b nf", "all")
        print("Face mask: ", face_mask)

        batch, num_vertices, num_coors = vertices.shape
        _, num_faces, _ = faces.shape

        # Fill padding tokens with 0 to prevent gather idx errors
        face_without_pad = faces.masked_fill(~rearrange(face_mask, "b nf -> b nf 1"), 0)

        # Collect vertex coordinates for each face: b x nf x nv x c
        faces_vertices = repeat(face_without_pad, "b nf nv -> b nf nv c", c=num_coors)
        vertices = repeat(vertices, "b nv c -> b nf nv c", nf=num_faces)
        face_coords = vertices.gather(-2, faces_vertices.long())

        # Discretize face coordinates
        discrete_face_coords = discretize(
            face_coords,
            continuous_range=self.coor_continuous_range,
            num_discrete=self.num_discrete_coors,
        )
        print("Discrete face coordinates: ", discrete_face_coords)

        # Pad invalid faces with pad_id
        discrete_padded_coords = discrete_face_coords.masked_fill(
            ~rearrange(face_mask, "b nf -> b nf 1 1"),
            self.pad_id,
        )
        print("Discrete face coordinates: ", discrete_face_coords)

        # Convert mesh to sequence: batch x ntokens
        input_ids = discrete_padded_coords.reshape(batch, -1)
        attention_mask = (input_ids != self.pad_id).float()

        # Add <bos> and <eos> tokens
        placeholder = torch.ones_like(input_ids[:, [0]]) * self.pad_id  # batch x 1
        input_ids = torch.cat((placeholder, input_ids, placeholder), dim=1)
        attention_mask = torch.cat((placeholder, attention_mask, placeholder), dim=1)

        # Final outputs
        data_dict["input_ids"] = input_ids.long()  # batch x (nf * 3 * 3 + 2)
        data_dict["attention_mask"] = attention_mask.float()  # batch x (nf * 3 * 3 + 2)
        data_dict["codes"] = discrete_padded_coords.long()  # batch x nf * 3 * 3
        data_dict["discrete_face_coords"] = discrete_face_coords

        return data_dict

    def forward(self, data_dict: dict) -> dict:

        encoder_output = self.tokenize(data_dict)
        decoder_output = self.detokenize(
            input_ids=encoder_output["codes"],
        )
        data_dict.update(encoder_output)
        data_dict.update(decoder_output)
        return data_dict
