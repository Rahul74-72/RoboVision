import numpy as np


def normalize_geometry(geometry, mean, std):
    """Return an L2-normalized geometry vector with safe std handling."""
    geometry = np.asarray(geometry, dtype=np.float32)
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)

    if geometry.ndim != 1 or mean.ndim != 1 or std.ndim != 1:
        raise ValueError("geometry, mean, and std must be 1-D vectors")

    if geometry.shape != mean.shape or geometry.shape != std.shape:
        raise ValueError("geometry, mean, and std must have the same shape")

    if not (
        np.isfinite(geometry).all()
        and np.isfinite(mean).all()
        and np.isfinite(std).all()
    ):
        raise ValueError("geometry, mean, and std must contain finite values")

    if (std < 0).any():
        raise ValueError("std must contain non-negative values")

    valid_std = std > 1e-6
    # Widen before subtraction and norm calculation to avoid float32 overflow.
    geometry64 = geometry.astype(np.float64)
    mean64 = mean.astype(np.float64)
    std64 = std.astype(np.float64)
    z = np.zeros_like(geometry64)
    np.divide(
        geometry64 - mean64,
        std64,
        out=z,
        where=valid_std,
    )

    norm = np.linalg.norm(z)
    return (z / norm if norm > 1e-8 else z).astype(np.float32)
