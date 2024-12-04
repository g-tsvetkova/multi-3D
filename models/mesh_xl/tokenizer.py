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


def torch_lexical_sort(vertices: torch.Tensor) -> torch.Tensor:
    """
    Sort vertices by z, y, x coordinates lexicographically using PyTorch.

    Args:
        vertices: Tensor of shape (batch_size, nv, 3)

    Returns:
        sorted_vertices: Tensor of shape (batch_size, nv, 3) with sorted vertices
    """
    sorted_vertices = torch.empty_like(vertices)  # Placeholder for sorted vertices

    for i, batch in enumerate(vertices):  # Process each batch independently
        # Sort by x first, then y, then z to achieve z > y > x lexicographical sort
        sorted_indices = torch.argsort(batch[:, 0], stable=True)  # Sort by x
        batch = batch[sorted_indices]

        sorted_indices = torch.argsort(batch[:, 1], stable=True)  # Sort by y within x
        batch = batch[sorted_indices]

        sorted_indices = torch.argsort(batch[:, 2], stable=True)  # Sort by z within y
        sorted_vertices[i] = batch[sorted_indices]  # Store the sorted batch

    return sorted_vertices


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

    @staticmethod
    def reorder_vertices(vertices: Tensor) -> Tensor:
        """
        Reorder vertices by z, y, x.
        """
        # Sort vertices by z, y, x
        sorted_vertices = torch_lexical_sort(vertices)
        return sorted_vertices

    @staticmethod
    def reorder_faces(vertices: Tensor, faces: Tensor) -> Tensor:
        """
        Cyclically permute face vertices by z, y, x and reorder faces.
        """
        # Gather face coordinates
        face_coords = vertices[faces]

        # Cyclically permute each face by z, y, x order
        for i in range(face_coords.size(0)):
            order = torch.lexsort(
                (face_coords[i, :, 0], face_coords[i, :, 1], face_coords[i, :, 2])
            )
            faces[i] = faces[i][order]

        # Sort faces globally based on the centroid (z, y, x)
        face_centroids = face_coords.mean(dim=1)
        face_order = torch.lexsort(
            (face_centroids[:, 0], face_centroids[:, 1], face_centroids[:, 2])
        )
        return faces[face_order]

    def tokenize(self, data_dict: dict) -> dict:
        """
        Turn 3D meshes into sequential tokens: <bos> [<x>, <y>, <z>], ... <eos>.
        """
        # Extract vertices and faces
        vertices = data_dict["vertices"]  # shape: batch x nv x 3
        faces = data_dict["faces"]  # shape: batch x nf x 3

        # Preprocessing: normalize, reorder vertices, and reorder faces
        vertices = self.normalize_vertices(vertices)

        # Reorder vertices by (z, y, x) using lexicographical sorting
        vertices = self.reorder_vertices(vertices)

        # Reorder faces by (z, y, x) using lexicographical sorting
        faces = self.reorder_faces(vertices, faces)

        # Generate face mask
        face_mask = reduce(faces != self.pad_id, "b nf c -> b nf", "all")

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

        # Pad invalid faces with pad_id
        discrete_padded_coords = discrete_face_coords.masked_fill(
            ~rearrange(face_mask, "b nf -> b nf 1 1"),
            self.pad_id,
        )

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
