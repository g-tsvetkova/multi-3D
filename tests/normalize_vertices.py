import torch


def normalize_vertices(vertices: torch.Tensor) -> torch.Tensor:
    """
    Normalize vertices into a unit cube based on the longest axis.
    """
    min_coords = vertices.min(dim=0).values
    print("Min Coord:", min_coords)
    max_coords = vertices.max(dim=0).values
    print("Max Coord:", max_coords)
    center = (min_coords + max_coords) / 2
    print("Center:", center)
    longest_axis = (max_coords - min_coords).max()
    print("Longest axis: ", longest_axis)
    return (vertices - center) / longest_axis


# from Polygen
def normalize_vertices_scale_old(vertices: torch.Tensor) -> torch.Tensor:
    """Scale vertices so that the long diagonal of the bounding box is one

    Args:
        vertices: unscaled vertices of shape (num_vertices, 3)
    Returns:
        scaled_vertices: scaled vertices of shape (num_vertices, 3)
    """
    vert_min, _ = torch.min(vertices, dim=0)
    print("Min Coord:", vert_min)
    vert_max, _ = torch.max(vertices, dim=0)
    print("Max Coord:", vert_max)
    extents = vert_max - vert_min
    print("Extents:", extents)
    scale = torch.sqrt(torch.sum(extents**2))
    print("Scale:", scale)
    scaled_vertices = vertices / scale
    return scaled_vertices


def normalize_vertices_scale(vertices: torch.Tensor) -> torch.Tensor:
    """Scale vertices so that the long diagonal of the bounding box is one.

    Args:
        vertices: unscaled vertices of shape (batch_size, num_vertices, 3)
    Returns:
        scaled_vertices: scaled vertices of shape (batch_size, num_vertices, 3)
    """
    # Compute min and max per batch along the vertex dimension
    vert_min, _ = torch.min(vertices, dim=1, keepdim=True)  # Shape: (batch_size, 1, 3)
    vert_max, _ = torch.max(vertices, dim=1, keepdim=True)  # Shape: (batch_size, 1, 3)

    # Compute extents and scale
    extents = vert_max - vert_min  # Shape: (batch_size, 1, 3)
    scale = torch.sqrt(
        torch.sum(extents**2, dim=-1, keepdim=True)
    )  # Shape: (batch_size, 1, 1)

    # Normalize vertices
    scaled_vertices = vertices / scale  # Broadcasting scales appropriately
    return scaled_vertices


def test_vertex_normalization():
    # Test case 1: Simple cube
    test1 = torch.tensor(
        [
            [
                [0.3841, 0.2144, 0.3698],
                [0.3030, 0.7712, 0.4292],
                # [0.0000, 0.0000, 0.0000],
                # [0.0000, 0.0000, 0.0000],
                # [0.0000, 0.0000, 0.0000],
                # [0.0000, 0.0000, 0.0000],
                # [0.0000, 0.0000, 0.0000],
                # [0.0000, 0.0000, 0.0000],
                # [0.0000, 0.0000, 0.0000],
                # [0.0000, 0.0000, 0.0000],
            ]
        ]
    )
    # Test case 2: Irregular shape
    irregular_vertices = torch.tensor(
        [[2, 3, 1], [5, 1, 4], [0, 4, 2], [3, 0, 5]], dtype=torch.float32
    )

    # Test case 3: Mixed case with some zero vertices
    mixed_vertices = torch.tensor(
        [
            [0.4171, 0.3052, 0.3645],
            [0.3422, 0.9911, 0.3220],
            [0.9309, 0.0793, 0.6643],
            [0.5219, 0.2108, 0.0818],
            [0.0000, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.0000],
        ],
        dtype=torch.float32,
    )

    def print_vertex_info(name, vertices):
        print(f"\n{name} Vertices:")
        print("Original vertices:")
        print(vertices)
        print("\nNormalize_vertices result:")
        normalized = normalize_vertices_scale(vertices)
        print(normalized)
        print("\nNormalize_vertices_scale result:")
        scaled = normalize_vertices_scale(vertices)
        print(scaled)

        # Additional checks
        print("\nChecks:")
        print(
            f"Original bbox size: {vertices.max(dim=0).values - vertices.min(dim=0).values}"
        )
        print(
            f"Normalized bbox size (normalize_vertices): {normalized.max(dim=0).values - normalized.min(dim=0).values}"
        )
        print(
            f"Normalized bbox size (normalize_vertices_scale): {scaled.max(dim=0).values - scaled.min(dim=0).values}"
        )

    # Run tests
    print_vertex_info("Cube", test1)
    print_vertex_info("Irregular", irregular_vertices)
    print_vertex_info("Mixed", mixed_vertices)


# Run the test
test_vertex_normalization()
