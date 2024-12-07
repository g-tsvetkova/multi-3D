import torch

torch.set_default_device("cpu")  # Force CPU usage


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


def demonstrate_sorting_and_reindexing():
    # Create a sample vertices and faces tensor with multiple batches
    vertices = torch.randn(2, 4, 3)  # 3 batches, 10 vertices each
    faces = torch.randint(0, 4, (2, 3, 3))  # 3 batches, 5 faces each, vertex indices

    print("Original Vertices:\n", vertices)
    print("\nOriginal Faces:\n", faces)

    # Sort vertices lexicographically
    sorted_vertices, sorting_indices = torch_lexical_sort(vertices)

    print("\nSorted Vertices:\n", sorted_vertices)
    print("\nSorting Indices:\n", sorting_indices)

    # Reindex faces
    reindexed_faces = reindex_faces_after_sort(faces, sorting_indices)

    print("\nReindexed Faces:\n", reindexed_faces)


# Uncomment to run demonstration
demonstrate_sorting_and_reindexing()


