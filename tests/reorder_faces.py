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
        sorted_vertices[i] = batch[sorted_indices]  # Store the sorted batch

    return sorted_vertices


def reorder_vertices_and_faces(vertices: torch.Tensor, faces: torch.Tensor):
    """
    Reorder vertices by z, y, x coordinates and reindex faces.

    Args:
        vertices: Tensor of shape (batch_size, nv, 3)
        faces: Tensor of shape (batch_size, nf, 3)

    Returns:
        sorted_vertices: Tensor of shape (batch_size, nv, 3) with sorted vertices
        reindexed_faces: Tensor of shape (batch_size, nf, 3) with updated face indices
    """
    sorted_vertices = torch.empty_like(vertices)  # Placeholder for sorted vertices
    reindexed_faces = torch.empty_like(faces)  # Placeholder for reindexed faces

    for i, (batch_vertices, batch_faces) in enumerate(
        zip(vertices, faces)
    ):  # Process each batch independently
        # Sort vertices and get the sorted indices
        sorted_indices = torch.argsort(batch_vertices[:, 0], stable=True)
        batch_vertices = batch_vertices[sorted_indices]

        sorted_indices = torch.argsort(batch_vertices[:, 1], stable=True)
        batch_vertices = batch_vertices[sorted_indices]

        sorted_indices = torch.argsort(batch_vertices[:, 2], stable=True)
        batch_vertices = batch_vertices[sorted_indices]

        sorted_vertices[i] = batch_vertices

        # Create a mapping from old indices to new indices
        index_mapping = torch.argsort(sorted_indices)

        # Reindex the faces using the mapping
        reindexed_faces[i] = index_mapping[batch_faces]

    return sorted_vertices, reindexed_faces


def test_reorder_vertices_and_faces():
    """
    Test the reorder_vertices_and_faces function with example input.
    """

    # Example vertices (nv x 3)
    vertices = torch.tensor(
        [
            [0.0, 0.0, 0.0],  # 0
            [1.0, 0.0, 0.0],  # 1
            [0.0, 1.0, 0.0],  # 2
            [1.0, 1.0, 1.0],  # 3
            [0.0, 0.0, 1.0],  # 4
        ],
        dtype=torch.float32,
    ).unsqueeze(
        0
    )  # Add batch dimension (1 x nv x 3)

    # Example faces (nf x 3)
    faces = torch.tensor(
        [[0, 1, 2], [2, 3, 4], [1, 2, 4]], dtype=torch.int64
    ).unsqueeze(
        0
    )  # Add batch dimension (1 x nf x 3)

    # Expected output (sorted by z, y, x)
    expected_sorted_vertices = torch.tensor(
        [
            [0.0, 0.0, 0.0],  # 0
            [1.0, 0.0, 0.0],  # 1
            [0.0, 1.0, 0.0],  # 2
            [0.0, 0.0, 1.0],  # 4
            [1.0, 1.0, 1.0],  # 3
        ],
        dtype=torch.float32,
    ).unsqueeze(0)

    # Expected reindexed faces
    # expected faces [0, 1, 2] - > [0, 1, 2], [2, 3, 4] -> [2, 4, 3], [1, 2, 4] - >[1, 2, 3]

    expected_reindexed_faces = torch.tensor(
        [[0, 1, 2], [2, 4, 3], [1, 2, 3]], dtype=torch.int64
    ).unsqueeze(0)

    # Run the reorder_vertices_and_faces function
    sorted_vertices, reindexed_faces = reorder_vertices_and_faces(vertices, faces)

    # Check if the result matches the expected output
    assert torch.equal(
        sorted_vertices, expected_sorted_vertices
    ), f"Test failed for vertices! \nExpected:\n{expected_sorted_vertices}\nGot:\n{sorted_vertices}"

    assert torch.equal(
        reindexed_faces, expected_reindexed_faces
    ), f"Test failed for faces! \nExpected:\n{expected_reindexed_faces}\nGot:\n{reindexed_faces}"

    print("Test passed! Sorted vertices and reindexed faces are correct.")


# Run the test
test_reorder_vertices_and_faces()
