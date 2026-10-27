"""Lifetime and cold-compilation guards for directly registered CPU weights."""

from contextlib import contextmanager
import weakref

import torch

from magi_compiler.cuda.cudart import pin_memory_in_place, unpin_memory_in_place


_owners = []
_suspend_depth = 0


def register_offload_owner(owner, device, pin=True):
    # Only weak references: the model owns the CPU storage and its registration.
    _owners.append((weakref.ref(owner), device, pin))


def _live_owners():
    live = [(ref(), device, pin) for ref, device, pin in _owners]
    live = [(owner, device, pin) for owner, device, pin in live if owner is not None]
    _owners[:] = [(weakref.ref(owner), device, pin) for owner, device, pin in live]
    return live


def is_offload_weight(value):
    if value.device.type != "cpu" or not value.numel():
        return False
    start = value.data_ptr()
    end = start + value.numel() * value.element_size()
    return any(owner.data_ptr() <= start and end <= owner.data_ptr() + owner.numel() * owner.element_size()
               for owner, _, _ in _live_owners())


def pin_offload_weights():
    if _suspend_depth:
        return
    for owner, device, pin in _live_owners():
        if not pin:
            continue
        with torch.cuda.device(device):
            pin_memory_in_place(owner)


@contextmanager
def suspend_offload_registration(synchronize_ranks=False):
    """Load/execute cold kernels before registering the selected CPU storage.

    All ranks must enter collectively when synchronize_ranks is enabled, just
    as required by the offload backend's cold-compilation barrier.
    """
    global _suspend_depth
    outermost = _suspend_depth == 0
    _suspend_depth += 1
    group = None
    completed = False
    try:
        if outermost:
            if synchronize_ranks and torch.distributed.is_initialized():
                from magi_compiler.utils.dist_utils import get_cpu_gloo_group
                group = get_cpu_gloo_group()
                if group is None:
                    raise RuntimeError("Deferred offload registration requires a CPU/gloo group")
            for device in {device for _, device, _ in _live_owners()}:
                torch.cuda.synchronize(device)
            if group is not None:
                torch.distributed.barrier(group=group)
            for owner, device, pin in _live_owners():
                if not pin:
                    continue
                with torch.cuda.device(device):
                    unpin_memory_in_place(owner)
            if group is not None:
                torch.distributed.barrier(group=group)
        yield
        if outermost:
            for device in {device for _, device, _ in _live_owners()}:
                torch.cuda.synchronize(device)
            if group is not None:
                torch.distributed.barrier(group=group)
        completed = True
    finally:
        _suspend_depth -= 1
        # A failed compilation/synchronization may leave DMA in flight. Do not
        # re-register memory or enter another collective while unwinding it.
        if outermost and completed:
            pin_offload_weights()
            if group is not None:
                torch.distributed.barrier(group=group)
