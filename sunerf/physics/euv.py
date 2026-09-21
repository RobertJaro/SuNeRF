"""Numerical helpers shared by EUV plasma models and renderers.

Ray integrations and diagnostic temperature histograms are evaluated on nodes
rather than cell centres.  Keeping the node-weight calculation in one place
prevents quadrature conventions from drifting apart, especially for
non-uniform and hierarchically sampled grids.
"""

from __future__ import annotations

import numpy as np
import torch

from sunerf.response.numerics import trapezoid_node_weights


R_SUN_CM = 6.957e10


def numpy_trapezoid_node_weights(nodes) -> np.ndarray:
    """Backward-compatible public alias for response quadrature."""
    return trapezoid_node_weights(nodes)


def torch_trapezoid_node_weights(nodes: torch.Tensor) -> torch.Tensor:
    """Torch equivalent of :func:`numpy_trapezoid_node_weights`.

    The final dimension contains the integration nodes; arbitrary batch
    dimensions are retained.  This function is differentiable with respect to
    the nodes, which is useful when hierarchical ray locations are merged.
    """
    if nodes.ndim < 1 or nodes.shape[-1] < 2:
        raise ValueError("quadrature nodes must have at least two entries on the final axis")
    if not torch.isfinite(nodes).all():
        raise ValueError("quadrature nodes contain non-finite values")

    spacing = nodes[..., 1:] - nodes[..., :-1]
    # Coincident nodes are zero-width intervals and contribute nothing. They
    # occur when a deterministic hierarchical sample lands on a coarse node.
    if torch.any(spacing < 0):
        raise ValueError("quadrature nodes must be non-decreasing")

    middle = 0.5 * (spacing[..., :-1] + spacing[..., 1:])
    return torch.cat((0.5 * spacing[..., :1], middle, 0.5 * spacing[..., -1:]), dim=-1)
