"""Deterministic parameter partitioning before CPU host registration."""

import math


def choose_offload_weights(parameters, resident_ratio):
    """Return names outside the GPU budget, using whole parameter tensors.

    Small tensors are kept first so routing/norm parameters remain resident
    whenever possible. Ties use parameter names for repeatable rank-local plans.
    CPU backups of resident parameters are not registered. Buffers are excluded.
    """
    if not math.isfinite(resident_ratio) or not 0.0 <= resident_ratio <= 1.0:
        raise ValueError("gpu_resident_weight_ratio must be finite and between 0 and 1")
    sizes = {name: value.numel() * value.element_size() for name, value in parameters.items()}
    budget = int(sum(sizes.values()) * resident_ratio)
    resident_bytes = 0
    offloaded = set()
    for name, size in sorted(sizes.items(), key=lambda item: (item[1], item[0])):
        if resident_bytes + size <= budget:
            resident_bytes += size
        else:
            offloaded.add(name)
    return offloaded
