from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.fx import Graph, GraphModule

from magi_compiler import _api
from magi_compiler.config import CompileConfig, OffloadPolicy
from magi_compiler.offload import host_weights
from magi_compiler.offload.scheduler import OffloadRuntimeContext, ResidentRatioScheduler
from magi_compiler.offload.weight_layout import choose_offload_weights


@pytest.fixture(autouse=True)
def isolated_owners(monkeypatch):
    monkeypatch.setattr(host_weights, "_owners", [])
    monkeypatch.setattr(host_weights, "_suspend_depth", 0)


@pytest.mark.parametrize("ratio,count", [(0.0, 10), (0.6, 4), (1.0, 0)])
def test_ratio_endpoints_and_exact_byte_budget(ratio, count):
    parameters = {f"w{i}": torch.ones(8) for i in range(10)}
    selected = choose_offload_weights(parameters, ratio)
    assert len(selected) == count
    assert selected == choose_offload_weights(dict(reversed(list(parameters.items()))), ratio)


def test_partition_counts_bytes_and_keeps_small_parameters_first():
    # 4 bytes + 8 bytes + 12 bytes = 24, resident budget = 12.
    parameters = {"large": torch.ones(3, dtype=torch.float32),
                  "small": torch.ones(2, dtype=torch.bfloat16),
                  "middle": torch.ones(4, dtype=torch.bfloat16)}
    assert choose_offload_weights(parameters, 0.5) == {"large"}
    assert not choose_offload_weights({}, 0.6)


@pytest.mark.parametrize("ratio", [-0.01, 1.01, float("nan"), float("inf")])
def test_invalid_ratio_rejected(ratio):
    with pytest.raises(ValueError, match="between 0 and 1"):
        choose_offload_weights({"w": torch.ones(2)}, ratio)


@pytest.mark.parametrize("ratio,selected_names", [
    (0.0, {"small", "middle", "large"}), (0.5, {"large"}), (1.0, set()),
])
def test_only_offload_storage_is_registered(monkeypatch, ratio, selected_names):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.small = nn.Parameter(torch.arange(2, dtype=torch.bfloat16))
            self.middle = nn.Parameter(torch.arange(4, dtype=torch.bfloat16))
            self.large = nn.Parameter(torch.arange(3, dtype=torch.float32))
            self.register_buffer("buffer", torch.ones(4))

    config = CompileConfig()
    config.offload_config.offload_policy = OffloadPolicy.RESIDENT_RATIO
    config.offload_config.gpu_resident_weight_ratio = ratio
    config.offload_config.defer_weight_pinning = True
    model = Model()
    expected = {name: value.clone() for name, value in model.state_dict().items()}
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(host_weights, "pin_memory_in_place", lambda _: pytest.fail("premature registration"))
    _api._patch_cpu_offload_apply(Model, config)
    def convert(value):
        return value
    convert.__qualname__ = "Module.cuda.<locals>.<lambda>"
    model._apply(convert)
    actual = {name for name, value in model.named_parameters() if host_weights.is_offload_weight(value)}
    assert actual == selected_names
    assert not host_weights.is_offload_weight(model.buffer)
    expected_bytes = sum(value.numel() * value.element_size()
                         for name, value in model.named_parameters() if name in selected_names)
    registered_bytes = sum(owner.numel() * owner.element_size() for owner, _ in host_weights._live_owners())
    assert registered_bytes == expected_bytes
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, expected[name])


def make_scheduler(lookahead=2):
    graph = Graph()
    x = graph.placeholder("x")
    weights = [graph.placeholder(f"w{i}") for i in range(3)]
    for node in weights:
        node.meta["example_value"] = nn.Parameter(torch.ones(2))
    modules = []
    value = x
    for index in range(4):
        value = graph.call_module(f"submod_{index}", (value, weights[min(index, 2)]))
        modules.append(value)
    graph.output(value)
    config = CompileConfig()
    config.offload_config.offload_policy = OffloadPolicy.RESIDENT_RATIO
    config.offload_config.max_prefetch_lookahead = lookahead
    scheduler = ResidentRatioScheduler(config, modules,
        {node.name: [node.args[1].name] for node in modules}, {n.name: n for n in weights}, {})
    scheduler.select_weights({"w1", "w2"})
    return scheduler, modules, weights


def test_fixed_partition_survives_profiling_and_reentry():
    scheduler, _, _ = make_scheduler()
    schedule = dict(scheduler.prefetch_schedule)
    scheduler.schedule_kept_weights({"h2d_bandwidth": 0.0})
    scheduler.select_weights({"w1", "w2"})
    assert scheduler.kept_weights == {"w0"}
    assert scheduler.offloaded_weights == {"w1", "w2"}
    assert dict(scheduler.prefetch_schedule) == schedule


@pytest.mark.parametrize("lookahead", [0, 2])
def test_transfer_ready_wait_and_shared_weight(monkeypatch, lookahead):
    scheduler, modules, weights = make_scheduler(lookahead)
    events = []
    class Event:
        def record(self, stream):
            events.append("ready")
    class Stream:
        def wait_event(self, event):
            events.append("wait")
    def copy(value):
        events.append("dma")
        return SimpleNamespace(device=SimpleNamespace(type="cuda"))
    monkeypatch.setattr(torch.cuda, "stream", lambda _: nullcontext())
    monkeypatch.setattr(torch.cuda, "Event", Event)
    ctx = OffloadRuntimeContext({node: torch.ones(2) for node in weights},
                               Stream(), Stream(), {}, {}, transfer=copy)
    for node in modules:
        scheduler.prefetch(node.name, ctx)
        events.append("compute")
    # Two selected weights, including w2 used by two consumers: one DMA each.
    assert events.count("dma") == 2
    assert events.count("ready") == 2
    assert events.count("wait") == 3
    assert events.index("dma") < events.index("ready") < events.index("wait")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA/HIP device")
def test_residency_shared_between_entries_and_stage_reload():
    from magi_compiler.offload.offload_warpper import OffloadExecutor, release_offload_caches
    owner = torch.randn(32 * 32)
    host_weights.register_offload_owner(owner, torch.cuda.current_device())
    weights = [nn.Parameter(torch.randn(32, 32)), nn.Parameter(owner.view(32, 32))]
    graph = Graph()
    x = graph.placeholder("x")
    nodes = [graph.placeholder(f"w{i}") for i in range(2)]
    root = nn.Module()
    class Matmul(nn.Module):
        def forward(self, value, weight):
            return value @ weight
    value = x
    for index, (node, weight) in enumerate(zip(nodes, weights)):
        node.meta["example_value"] = weight
        root.add_module(f"submod_{index}", Matmul())
        value = graph.call_module(f"submod_{index}", (value, node))
    graph.output(value)
    config = CompileConfig()
    config.offload_config.offload_policy = OffloadPolicy.RESIDENT_RATIO
    config.offload_config.defer_weight_pinning = True
    entries = [OffloadExecutor(GraphModule(root, graph), config) for _ in range(2)]
    for entry in entries:
        entry.profiler.summarize = lambda: pytest.fail("fixed residency must not profile/repartition")
    inputs = torch.randn(4, 32, device="cuda")
    expected = inputs @ weights[0].cuda() @ weights[1].cuda()
    for entry in entries:
        with host_weights.suspend_offload_registration():
            assert not owner.is_pinned()
            torch.testing.assert_close(entry(inputs, *weights), expected)
        assert owner.is_pinned()
    assert entries[0].persistent_weights["w0"] is entries[1].persistent_weights["w0"]
    for _ in range(3):
        for entry in entries:
            torch.testing.assert_close(entry(inputs, *weights), expected)
            assert entry.scheduler.kept_weights == {"w0"}
    assert release_offload_caches() >= 2
    for entry in entries:
        assert not entry.persistent_weights
        torch.testing.assert_close(entry(inputs, *weights), expected)


def test_nested_cold_load_unpins_before_load_and_repins_after(monkeypatch):
    owner = torch.ones(8)
    host_weights.register_offload_owner(owner, 0)
    events = []
    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _: events.append("sync"))
    monkeypatch.setattr(host_weights, "pin_memory_in_place", lambda _: events.append("pin"))
    monkeypatch.setattr(host_weights, "unpin_memory_in_place", lambda _: events.append("unpin"))
    with host_weights.suspend_offload_registration():
        events.append("load")
        with host_weights.suspend_offload_registration():
            host_weights.pin_offload_weights()
            events.append("execute")
        assert "pin" not in events
    assert events == ["sync", "unpin", "load", "execute", "sync", "pin"]


def test_failed_cold_load_does_not_repin_or_mask_failure(monkeypatch):
    owner = torch.ones(8)
    host_weights.register_offload_owner(owner, 0)
    events = []
    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _: events.append("sync"))
    monkeypatch.setattr(host_weights, "pin_memory_in_place", lambda _: events.append("pin"))
    monkeypatch.setattr(host_weights, "unpin_memory_in_place", lambda _: events.append("unpin"))
    with pytest.raises(RuntimeError, match="cold load failed"):
        with host_weights.suspend_offload_registration():
            raise RuntimeError("cold load failed")
    assert events == ["sync", "unpin"]
    assert host_weights._suspend_depth == 0
    # A subsequent successful cold entry can recover registration.
    with host_weights.suspend_offload_registration():
        pass
    assert events[-3:] == ["unpin", "sync", "pin"]


def test_collective_registration_waits_for_cold_execution(monkeypatch):
    from magi_compiler.utils import dist_utils

    owner = torch.ones(8)
    host_weights.register_offload_owner(owner, 0)
    events = []
    group = object()
    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _: events.append("sync"))
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(dist_utils, "get_cpu_gloo_group", lambda: group)
    def barrier(*, group):
        events.append("barrier")
    monkeypatch.setattr(torch.distributed, "barrier", barrier)
    monkeypatch.setattr(host_weights, "pin_memory_in_place", lambda _: events.append("pin"))
    monkeypatch.setattr(host_weights, "unpin_memory_in_place", lambda _: events.append("unpin"))
    with host_weights.suspend_offload_registration(synchronize_ranks=True):
        events.append("execute")
    assert events == ["sync", "barrier", "unpin", "barrier", "execute",
                      "sync", "barrier", "pin", "barrier"]


