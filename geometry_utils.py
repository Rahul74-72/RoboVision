import numpy as np


def normalize_geometry(geometry, mean, std):
    """Return an L2-normalized geometry vector with safe std handling."""
    geometry = np.asarray(geometry, dtype=np.float32)
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)

    valid_std = np.abs(std) > 1e-6
    z = np.zeros_like(geometry, dtype=np.float32)
    np.divide(
        geometry - mean,
        std,
        out=z,
        where=valid_std,
    )

    norm = np.linalg.norm(z)
    return z / norm if norm > 1e-8 else z
