"""NumPy-only numerical helpers for response construction."""

import numpy as np


def trapezoid_node_weights(nodes) -> np.ndarray:
    """Return trapezoidal quadrature weights for strictly increasing nodes."""
    nodes = np.asarray(nodes)
    if nodes.ndim != 1 or nodes.size < 2:
        raise ValueError(
            "quadrature nodes must be a one-dimensional array with at least two entries"
        )
    if not np.isfinite(nodes).all():
        raise ValueError("quadrature nodes contain non-finite values")
    spacing = np.diff(nodes)
    if np.any(spacing <= 0):
        raise ValueError("quadrature nodes must be strictly increasing")
    weights = np.empty_like(nodes, dtype=np.result_type(nodes.dtype, np.float32))
    weights[0] = 0.5 * spacing[0]
    weights[-1] = 0.5 * spacing[-1]
    if nodes.size > 2:
        weights[1:-1] = 0.5 * (spacing[:-1] + spacing[1:])
    return weights
