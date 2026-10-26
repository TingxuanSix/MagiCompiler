# Copyright (c) 2026 SandAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import ctypes
import ctypes.util
import weakref

import torch

_cudart = None


def init_cudart():
    global _cudart
    if _cudart is not None:
        return _cudart
    if torch.version.hip is not None:
        candidates = [ctypes.util.find_library("amdhip64"), "libamdhip64.so"]
    else:
        candidates = [
            ctypes.util.find_library("cudart"), "libcudart.so",
            "libcudart.so.11.0", "libcudart.so.12", "libcudart.so.13",
        ]
    for lib in candidates:
        if lib is None:
            continue
        try:
            _cudart = ctypes.CDLL(lib)
            return _cudart
        except OSError:
            continue
    return None


def pin_memory_in_place(tensor: torch.Tensor):
    """
    Register existing CPU storage with CUDA/HIP, without allocating host memory
    through the accelerator allocator. The owner must outlive all tensor views.
    """
    if tensor.is_cuda or tensor.numel() == 0:
        return tensor
    # Host registration before PyTorch initializes its device context is not
    # recognized by is_pinned() on the HCU runtime.
    torch.cuda.init()
    if tensor.is_pinned():
        return tensor
    cudart = init_cudart()
    if cudart is None:
        raise RuntimeError("Cannot register offload weights: CUDA/HIP runtime library not found")

    ptr = tensor.data_ptr()
    size = tensor.numel() * tensor.element_size()
    prefix = "hip" if torch.version.hip is not None else "cuda"
    register = getattr(cudart, prefix + "HostRegister")
    unregister = getattr(cudart, prefix + "HostUnregister")
    res = register(ctypes.c_void_p(ptr), ctypes.c_size_t(size), 0)

    # 0: success, 712: already registered; only unregister storage we own.
    if res not in (0, 712):
        raise RuntimeError(f"{prefix}HostRegister failed with error code {res}")
    if res == 0:
        tensor._magi_host_registration = weakref.finalize(tensor, unregister, ctypes.c_void_p(ptr))
    return tensor


def unpin_memory_in_place(tensor: torch.Tensor):
    """Undo a registration owned by MagiCompiler, preserving the CPU storage."""
    registration = getattr(tensor, "_magi_host_registration", None)
    if registration is not None and registration.alive:
        _, unregister, args, kwargs = registration.peek()
        result = unregister(*args, **kwargs)
        if result != 0:
            raise RuntimeError(f"HostUnregister failed with error code {result}")
        registration.detach()
