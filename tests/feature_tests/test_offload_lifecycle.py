"""Offload correctness at cold entry and across pipeline stage eviction."""

import pytest
import torch
from torch import nn
from torch.fx import Graph, GraphModule

from magi_compiler.config import CompileConfig, OffloadPolicy
from magi_compiler.offload.offload_warpper import OffloadExecutor, release_offload_caches
from magi_compiler.cuda.cudart import pin_memory_in_place


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA/HIP device")
def test_in_place_host_registration():
    master = torch.arange(4096, dtype=torch.float32)
    pointer = master.data_ptr()
    assert pin_memory_in_place(master) is master
    assert master.data_ptr() == pointer
    assert master.is_pinned()
    # Re-registration is a no-op and must not install a second unregister.
    assert pin_memory_in_place(master) is master
    result = master.to("cuda", non_blocking=True)
    torch.cuda.synchronize()
    torch.testing.assert_close(result.cpu(), master)


class Matmul(nn.Module):
    def forward(self, x, weight):
        return x @ weight


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA/HIP device")
@pytest.mark.parametrize("policy", [OffloadPolicy.BASE, OffloadPolicy.HEURISTIC])
@pytest.mark.parametrize("lookahead", [0, 2])
def test_stage_release_and_reload(policy, lookahead):
    graph = Graph()
    x = graph.placeholder("x")
    weights = [nn.Parameter(torch.randn(32, 32).pin_memory()) for _ in range(3)]
    nodes = [graph.placeholder(f"w{i}") for i in range(3)]
    for node, weight in zip(nodes, weights):
        node.meta["example_value"] = weight
    root = nn.Module()
    value = x
    for i, weight in enumerate(nodes):
        root.add_module(f"submod_{i}", Matmul())
        value = graph.call_module(f"submod_{i}", (value, weight))
    graph.output(value)
    config = CompileConfig()
    config.offload_config.model_cpu_offload = True
    config.offload_config.offload_policy = policy
    config.offload_config.max_prefetch_lookahead = lookahead
    config.offload_config.gpu_resident_weight_ratio = 0.8
    executor = OffloadExecutor(GraphModule(root, graph), config)
    executor.profiler.get_h2d_bandwidth = lambda: 20.0
    inputs = torch.randn(4, 32, device="cuda")
    expected = inputs
    for weight in weights:
        expected = expected @ weight.cuda()
    for _ in range(4):
        torch.testing.assert_close(executor(inputs, *weights), expected)
    schedule = dict(getattr(executor.scheduler, "prefetch_schedule", {}))
    assert release_offload_caches() >= 1
    assert not executor.persistent_weights
    assert not executor.submod_0_weight_handoff
    for _ in range(2):
        torch.testing.assert_close(executor(inputs, *weights), expected)
    assert dict(getattr(executor.scheduler, "prefetch_schedule", {})) == schedule


