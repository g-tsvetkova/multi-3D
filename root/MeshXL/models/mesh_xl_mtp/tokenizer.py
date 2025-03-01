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


class MeshTokenizer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.pad_id = self.codebook_size +2    #-1
        self.num_discrete_coors = args.n_discrete_size  # default: 128
        self.codebook_size = args.n_discrete_size  # default: 128
        self.coor_continuous_range = (-1.0, 1.0)

    def tokenize(self, data_dict: dict) -> dict:
        """
        Turn 3D meshes into sequential tokens: <bos> [<x>, <y>, <z>], ... <eos>.
        """
        # Extract vertices and faces
        vertices = data_dict["vertices"]  # shape: batch x nv x 3
        faces = data_dict["faces"]  # shape: batch x nf x 3
        # print("Raw vertices: ", vertices)
        # print("Raw faces: ", faces)

        # Generate face mask
        face_mask = reduce(faces != self.pad_id, "b nf c -> b nf", "all")
        # print("Face mask: ", face_mask)

        batch, num_vertices, num_coors = vertices.shape
        _, num_faces, _ = faces.shape

        # Fill padding tokens with 0 to prevent gather idx errors
        face_without_pad = faces.masked_fill(~rearrange(face_mask, "b nf -> b nf 1"), 0)

        # Collect vertex coordinates for each face: b x nf x nv x c
        faces_vertices = repeat(face_without_pad, "b nf nv -> b nf nv c", c=num_coors)
        vertices = repeat(vertices, "b nv c -> b nf nv c", nf=num_faces)
        # print(vertices.shape)
        # print(faces_vertices.shape)
        # print("max face index:", faces_vertices.max().item(), "num_vertices:", vertices.shape[1])
        # print("min face index:", faces_vertices.min().item())

        face_coords = vertices.gather(-2, faces_vertices.long())

        # Discretize face coordinates
        discrete_face_coords = discretize(
            face_coords,
            continuous_range=self.coor_continuous_range,
            num_discrete=self.num_discrete_coors,
        )
        # print("Discrete face coordinates: ", discrete_face_coords)

        # Pad invalid faces with pad_id
        discrete_padded_coords = discrete_face_coords.masked_fill(
            ~rearrange(face_mask, "b nf -> b nf 1 1"),
            self.pad_id,
        )
        #print("Discrete face coordinates: ", discrete_face_coords)

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
    
    def detokenize(self, input_ids: Tensor) -> dict:
        '''
        Turn sequential tokens: <bos> [<x>, <y>, <z>], ... <eos> into 3D meshes
        '''
        # input_ids: b (n q) or b n q, without <bos> or <eos>
        input_ids = input_ids.reshape(input_ids.shape[0], -1)
        # batch x nface
        face_mask = reduce(
            input_ids != self.pad_id, 'b (nf c) -> b nf', 'all', c = 9
        )
        
        # batch x (nface x 9) -> batch x nface x 3 x 3
        pred_face_coords = input_ids.reshape(input_ids.shape[0], -1, 9)
        pred_face_coords = rearrange(
            pred_face_coords, '... (v c) -> ... v c', v = 3
        )
        
        # back to continuous space
        continuous_coors = undiscretize(
            pred_face_coords,
            num_discrete = self.num_discrete_coors,
            continuous_range = self.coor_continuous_range
        )
        # mask padding coordinates out with nan
        continuous_coors = continuous_coors.masked_fill(
            ~rearrange(face_mask, 'b nf -> b nf 1 1'), 
            float('nan')
        )
        output_dict = {}
        output_dict['recon_faces'] = continuous_coors
        
        return output_dict

    def forward(self, data_dict: dict) -> dict:

        encoder_output = self.tokenize(data_dict)
        decoder_output = self.detokenize(
            input_ids=encoder_output["codes"],
        )
        data_dict.update(encoder_output)
        data_dict.update(decoder_output)
        return data_dict
