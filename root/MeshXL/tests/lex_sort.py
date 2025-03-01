import torch


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
        print(sorted_indices)
        sorted_vertices[i] = batch[sorted_indices]  # Store the sorted batch

    return sorted_vertices


# Test the function
vertices = torch.rand(2, 5, 3)  # Random vertices (batch x nv x 3)
print("Original vertices:")
print(vertices)

sorted_vertices = torch_lexical_sort(vertices)
print("Sorted vertices:")
print(sorted_vertices)
