"""Engine-level nvcomp enable/fallback policy.

``check_engine_nvcomp_enable`` is the single place that decides whether nvcomp
runs at all. Every unsupported configuration has to be declined *here*, in the
engine process, because the alternative failure mode is severe: a strategy that
refuses inside a spawned worker kills that worker, and the engine then waits
forever on a ``ready`` message that can never arrive. A server that hangs on
startup is much worse than one that transfers uncompressed.

These are pure policy checks -- no GPU, no nvcomp extension, no allocation --
so unlike the roundtrip tests in this directory they run anywhere.

The multi-group cell covers DSA models (DeepSeek-V4, GLM-5.3-Flash), which
split their layers into an MLA-latent group and an indexer group with different
geometry. Those per-group transfers bypass the compressor entirely.
"""

import pytest
import torch

from flexkv.transfer.compression.ans import ans_utils


class _Layout:
    def __init__(self, kv_dim=1, num_kv_heads=1, chunk_size=65536):
        self.kv_dim = kv_dim
        self.num_kv_heads = num_kv_heads
        self._chunk_size = chunk_size

    def get_chunk_size(self):
        # Callers multiply this by dtype.itemsize, so return elements.
        return self._chunk_size


class _Handle:
    def __init__(self, dtype=torch.bfloat16, **kw):
        self.dtype = dtype
        self.kv_layout = _Layout(**kw)


class _ModelConfig:
    def __init__(self, layer_groups=None):
        self.layer_groups = layer_groups


def _gpu_groups(dtype=torch.bfloat16, chunk_size=32768):
    # 32768 elements * 2 bytes = 65536 B, GLM-5.3's real latent chunk.
    return {0: [_Handle(dtype=dtype, chunk_size=chunk_size)]}


def _enable(monkeypatch, **kw):
    monkeypatch.setenv("FLEXKV_ENABLE_NVCOMP", "1")
    kw.setdefault("gpu_handle_groups", _gpu_groups())
    kw.setdefault("layerwise_enabled", False)
    kw.setdefault("cpu_handle", _Handle())
    return ans_utils.check_engine_nvcomp_enable(
        kw.pop("gpu_handle_groups"), **kw)


@pytest.mark.unit
def test_env_off_disables(monkeypatch):
    monkeypatch.setenv("FLEXKV_ENABLE_NVCOMP", "0")
    assert ans_utils.check_engine_nvcomp_enable(
        _gpu_groups(), layerwise_enabled=False, cpu_handle=_Handle()) is False


@pytest.mark.unit
def test_env_off_returns_before_any_guard(monkeypatch, caplog):
    """Guards the tests, not the code.

    With the env switch off the function returns immediately and logs nothing,
    so a cell that forgets to set it would assert ``result is False`` against
    the wrong branch and pass for the wrong reason. Two cells in this file did
    exactly that before this was pinned down.
    """
    monkeypatch.setenv("FLEXKV_ENABLE_NVCOMP", "0")
    with caplog.at_level("WARNING"):
        # Deliberately unsupported in three ways at once.
        ans_utils.check_engine_nvcomp_enable(
            _gpu_groups(dtype=torch.float32, chunk_size=1),
            layerwise_enabled=True,
            cpu_handle=_Handle(kv_dim=1, num_kv_heads=8),
            model_config=_ModelConfig([object(), object()]))
    assert caplog.text == "", (
        "env-off must short-circuit before every guard; if this logs, the "
        "fallback cells can no longer distinguish 'declined by the guard "
        "under test' from 'declined because the switch was off'")


@pytest.mark.unit
@pytest.mark.skipif(not ans_utils.AVAILABLE,
                    reason="extension not built with nvcomp")
def test_uniform_model_enables(monkeypatch):
    """The baseline: nothing unsupported, so nvcomp stays on.

    Without this the fallback cells below would pass even if the function
    returned False unconditionally.
    """
    assert _enable(monkeypatch, model_config=_ModelConfig(None)) is True


@pytest.mark.unit
@pytest.mark.skipif(not ans_utils.AVAILABLE,
                    reason="extension not built with nvcomp")
@pytest.mark.parametrize("ngroups", [2, 3])
def test_multi_group_falls_back_gracefully(monkeypatch, caplog, ngroups):
    """A DSA model must be declined here, and must NOT raise.

    Raising is the regression this guards: the refusal used to live in
    ``NvcompGpuCpuStrategy.attach()``, inside the worker process, where it
    killed the worker and hung the engine on startup.
    """
    groups = [object()] * ngroups
    with caplog.at_level("WARNING"):
        result = _enable(monkeypatch, model_config=_ModelConfig(groups))
    assert result is False
    assert "[nvcomp-fallback]" in caplog.text
    assert f"{ngroups} heterogeneous KV layer groups" in caplog.text


@pytest.mark.unit
@pytest.mark.skipif(not ans_utils.AVAILABLE,
                    reason="extension not built with nvcomp")
def test_missing_model_config_does_not_crash(monkeypatch):
    """``model_config`` is optional; absence must not be read as multi-group."""
    assert _enable(monkeypatch, model_config=None) is True
    assert _enable(monkeypatch) is True


@pytest.mark.unit
@pytest.mark.skipif(not ans_utils.AVAILABLE,
                    reason="extension not built with nvcomp")
def test_empty_layer_groups_is_uniform(monkeypatch):
    """An empty list means "no groups", not "heterogeneous"."""
    assert _enable(monkeypatch, model_config=_ModelConfig([])) is True


@pytest.mark.unit
@pytest.mark.skipif(not ans_utils.AVAILABLE,
                    reason="extension not built with nvcomp")
def test_layerwise_falls_back(monkeypatch, caplog):
    with caplog.at_level("WARNING"):
        result = _enable(monkeypatch, layerwise_enabled=True,
                         model_config=_ModelConfig(None))
    assert result is False
    assert "[nvcomp-fallback]" in caplog.text


@pytest.mark.unit
@pytest.mark.skipif(not ans_utils.AVAILABLE,
                    reason="extension not built with nvcomp")
def test_packed_4d_layout_falls_back(monkeypatch, caplog):
    with caplog.at_level("WARNING"):
        result = _enable(monkeypatch,
                         cpu_handle=_Handle(kv_dim=1, num_kv_heads=8),
                         model_config=_ModelConfig(None))
    assert result is False
    assert "[nvcomp-fallback]" in caplog.text


@pytest.mark.unit
@pytest.mark.skipif(not ans_utils.AVAILABLE,
                    reason="extension not built with nvcomp")
def test_chunk_below_minimum_falls_back(monkeypatch, caplog):
    # 1024 elements * 2 bytes = 2048 B, below the 4096 B ANS floor.
    monkeypatch.setenv("FLEXKV_ENABLE_NVCOMP", "1")
    with caplog.at_level("WARNING"):
        result = ans_utils.check_engine_nvcomp_enable(
            _gpu_groups(chunk_size=1024),
            layerwise_enabled=False,
            cpu_handle=_Handle(),
            model_config=_ModelConfig(None))
    assert result is False
    assert "[nvcomp-fallback]" in caplog.text


@pytest.mark.unit
@pytest.mark.skipif(not ans_utils.AVAILABLE,
                    reason="extension not built with nvcomp")
def test_unsupported_dtype_falls_back(monkeypatch, caplog):
    monkeypatch.setenv("FLEXKV_ENABLE_NVCOMP", "1")
    with caplog.at_level("WARNING"):
        result = ans_utils.check_engine_nvcomp_enable(
            _gpu_groups(dtype=torch.float32, chunk_size=16384),
            layerwise_enabled=False,
            cpu_handle=_Handle(),
            model_config=_ModelConfig(None))
    assert result is False
    assert "[nvcomp-fallback]" in caplog.text


@pytest.mark.unit
@pytest.mark.skipif(not ans_utils.AVAILABLE,
                    reason="extension not built with nvcomp")
def test_multi_group_precedes_other_guards(monkeypatch, caplog):
    """A DSA model is declined even when nothing else is wrong.

    GLM-5.3's latent chunk is 65536 B and bf16 -- it clears every other guard
    in the function -- so multi-group is the only thing that can stop it.
    """
    with caplog.at_level("WARNING"):
        result = _enable(monkeypatch,
                         model_config=_ModelConfig([object(), object()]))
    assert result is False
    assert "heterogeneous KV layer groups" in caplog.text
