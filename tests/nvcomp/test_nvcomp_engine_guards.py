"""Engine-level nvcomp enable/fallback policy.

``check_engine_nvcomp_enable`` is the single place that decides whether nvcomp
runs at all. Every unsupported configuration has to be declined *here*, in the
engine process, because the alternative failure mode is severe: a strategy that
refuses inside a spawned worker kills that worker, and the engine then waits
forever on a ``ready`` message that can never arrive. A server that hangs on
startup is much worse than one that transfers uncompressed.

These are pure policy checks -- no GPU, no nvcomp extension, no allocation --
so unlike the roundtrip tests in this directory they run anywhere.

The multi-group cells cover DSA models (DeepSeek-V4, GLM-5.3), which split
their layers into an MLA-latent group and an indexer group with different
geometry. Those are *supported*, but only by compressing the latent group and
moving the rest uncompressed -- so what has to be pinned down here is both that
a well-shaped one is accepted and that each way the shape can go wrong is
declined rather than raised.
"""

import logging

import pytest
import torch

from flexkv.transfer.compression.ans import ans_utils


class _CaptureHandler(logging.Handler):
    """Collect messages off the FLEXKV logger.

    ``caplog`` cannot be used here: ``FlexkvLogger`` sets ``propagate = False``
    on the "FLEXKV" logger, so records never reach the root handler pytest
    installs and ``caplog.text`` stays empty. Asserting against it therefore
    passes no matter what the code logs -- which is what every message
    assertion in this file used to do. ``tests/test_transfer_trace.py`` hit the
    same thing and solved it the same way.
    """

    def __init__(self):
        super().__init__()
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())

    @property
    def text(self):
        return "\n".join(self.messages)


@pytest.fixture
def flexkv_log():
    logger = logging.getLogger("FLEXKV")
    handler = _CaptureHandler()
    previous = logger.level
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    try:
        yield handler
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous)


@pytest.mark.unit
def test_capture_handler_actually_captures(flexkv_log):
    """Guards the harness: a silent capture makes every message cell vacuous."""
    from flexkv.common.debug import flexkv_logger
    flexkv_logger.warning("capture-probe")
    assert "capture-probe" in flexkv_log.text


class _Layout:
    def __init__(self, kv_dim=1, num_kv_heads=1, chunk_size=36864,
                 tokens_per_block=64, num_layer=78):
        self.kv_dim = kv_dim
        self.num_kv_heads = num_kv_heads
        self.tokens_per_block = tokens_per_block
        self.num_layer = num_layer
        self._chunk_size = chunk_size

    def get_chunk_size(self):
        # Elements, not bytes -- callers multiply by dtype.itemsize.
        return self._chunk_size


class _Handle:
    def __init__(self, dtype=torch.bfloat16, **kw):
        self.dtype = dtype
        self.kv_layout = _Layout(**kw)


class _ModelConfig:
    def __init__(self, layer_groups=None, dtype=torch.bfloat16):
        self.layer_groups = layer_groups
        self.dtype = dtype


class _Group:
    """A ``LayerGroupSpec`` as far as this guard is concerned."""

    def __init__(self, *, num_layers=78, num_kv_heads=1, head_size=576,
                 dtype=torch.bfloat16, compress_ratio=1):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.dtype = dtype
        self.compress_ratio = compress_ratio


def _glm53_groups(**latent_overrides):
    """GLM-5.3's real pair: 78-layer bf16 MLA latent + fp8 indexer.

    Latent chunk = 64 tokens * 1 head * (512 + 64) * 2 B = 73728 B; the indexer
    is page-packed at compress_ratio=64. Only the latent is ever inspected --
    the indexer's shape is the *reason* it is not compressed, not something the
    guard has to accept.
    """
    latent = dict(num_layers=78, num_kv_heads=1, head_size=576,
                  dtype=torch.bfloat16, compress_ratio=1)
    latent.update(latent_overrides)
    return [
        _Group(**latent),
        _Group(num_layers=78, num_kv_heads=1, head_size=8448,
               dtype=torch.float8_e4m3fn, compress_ratio=64),
    ]


def _gpu_groups(dtype=torch.bfloat16, chunk_size=36864):
    # The registered GPU handle carries the *primary* layout (layer_groups is
    # None on it, so get_chunk_size() is callable), and on a DSA model that
    # layout is group 0's: 36864 * 2 B = 73728 B.
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
def test_env_off_returns_before_any_guard(monkeypatch, flexkv_log):
    """Guards the tests, not the code.

    With the env switch off the function returns immediately and logs nothing,
    so a cell that forgets to set it would assert ``result is False`` against
    the wrong branch and pass for the wrong reason. Two cells in this file did
    exactly that before this was pinned down.
    """
    monkeypatch.setenv("FLEXKV_ENABLE_NVCOMP", "0")
    # Deliberately unsupported in four ways at once.
    ans_utils.check_engine_nvcomp_enable(
        _gpu_groups(dtype=torch.float32, chunk_size=1),
        layerwise_enabled=True,
        cpu_handle=_Handle(kv_dim=1, num_kv_heads=8),
        model_config=_ModelConfig(_glm53_groups(dtype=torch.float32)))
    assert flexkv_log.text == "", (
        "env-off must short-circuit before every guard; if this logs, the "
        "fallback cells can no longer distinguish 'declined by the guard "
        "under test' from 'declined because the switch was off'")


@pytest.mark.unit
@pytest.mark.skipif(not ans_utils.AVAILABLE,
                    reason="extension not built with nvcomp")
def test_uniform_model_enables(monkeypatch):
    """The baseline: nothing unsupported, so nvcomp stays on.

    Without this the fallback cells below would pass even against a function
    that returned False unconditionally.
    """
    assert _enable(monkeypatch, model_config=_ModelConfig(None)) is True


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
def test_layerwise_falls_back(monkeypatch, flexkv_log):
    result = _enable(monkeypatch, layerwise_enabled=True,
                     model_config=_ModelConfig(None))
    assert result is False
    assert "[nvcomp-fallback]" in flexkv_log.text
    assert "layerwise" in flexkv_log.text


@pytest.mark.unit
@pytest.mark.skipif(not ans_utils.AVAILABLE,
                    reason="extension not built with nvcomp")
def test_packed_4d_layout_falls_back(monkeypatch, flexkv_log):
    result = _enable(monkeypatch,
                     cpu_handle=_Handle(kv_dim=1, num_kv_heads=8),
                     model_config=_ModelConfig(None))
    assert result is False
    assert "packed 4D layout" in flexkv_log.text


@pytest.mark.unit
@pytest.mark.skipif(not ans_utils.AVAILABLE,
                    reason="extension not built with nvcomp")
def test_chunk_below_minimum_falls_back(monkeypatch, flexkv_log):
    # 1024 elements * 2 bytes = 2048 B, below the 4096 B ANS floor.
    monkeypatch.setenv("FLEXKV_ENABLE_NVCOMP", "1")
    result = ans_utils.check_engine_nvcomp_enable(
        _gpu_groups(chunk_size=1024),
        layerwise_enabled=False,
        cpu_handle=_Handle(),
        model_config=_ModelConfig(None))
    assert result is False
    assert f"{ans_utils.MIN_CHUNK_BYTES}B ANS minimum" in flexkv_log.text


@pytest.mark.unit
@pytest.mark.skipif(not ans_utils.AVAILABLE,
                    reason="extension not built with nvcomp")
def test_unsupported_dtype_falls_back(monkeypatch, flexkv_log):
    monkeypatch.setenv("FLEXKV_ENABLE_NVCOMP", "1")
    result = ans_utils.check_engine_nvcomp_enable(
        _gpu_groups(dtype=torch.float32, chunk_size=16384),
        layerwise_enabled=False,
        cpu_handle=_Handle(),
        model_config=_ModelConfig(None))
    assert result is False
    assert "unsupported" in flexkv_log.text


# --------------------------------------------------------------------------
# Multi-group (DSA: DeepSeek-V4, GLM-5.3)
# --------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.skipif(not ans_utils.AVAILABLE,
                    reason="extension not built with nvcomp")
def test_glm53_shape_is_accepted(monkeypatch, flexkv_log):
    """The whole point of the change: GLM-5.3 must come back True.

    Every refusal cell below would pass against a function that declined all
    multi-group models -- which is exactly what this used to do -- so this is
    the cell that gives them meaning.
    """
    assert _enable(monkeypatch,
                   model_config=_ModelConfig(_glm53_groups())) is True
    # Says which group it took and that the others are not compressed, because
    # "nvcomp is on" reads very differently from "nvcomp is on for 90% of the
    # bytes" to someone looking at a ratio in the logs.
    assert "[nvcomp-multi-group]" in flexkv_log.text
    assert "1 other group(s) transfer uncompressed" in flexkv_log.text


@pytest.mark.unit
@pytest.mark.skipif(not ans_utils.AVAILABLE,
                    reason="extension not built with nvcomp")
def test_three_groups_are_accepted(monkeypatch, flexkv_log):
    """Two groups is GLM-5.3's shape, not the supported limit.

    What matters is that exactly one group is compressed, whatever the total
    -- DSv4 registers more than two.
    """
    groups = _glm53_groups()
    groups.append(_Group(head_size=8448, dtype=torch.float8_e4m3fn,
                         compress_ratio=4))
    assert _enable(monkeypatch, model_config=_ModelConfig(groups)) is True
    assert "2 other group(s) transfer uncompressed" in flexkv_log.text


@pytest.mark.unit
@pytest.mark.skipif(not ans_utils.AVAILABLE,
                    reason="extension not built with nvcomp")
def test_compressed_group_is_the_first_one(monkeypatch):
    """The accept path inspects group 0, not "whichever group happens to fit".

    Pinned because the worker-side strategy binds ``COMPRESSED_GROUP_INDEX``
    independently. If this check ever started scanning for a compressible
    group while the strategy kept taking group 0, a model whose group 0 was
    unsupported would be accepted here and then compressed there -- with the
    engine having no way to know.
    """
    groups = _glm53_groups(dtype=torch.float32)
    # Group 1 is untouched and looks perfectly compressible; only group 0 is
    # broken. A scanning implementation would return True here.
    groups[1] = _Group(dtype=torch.bfloat16, compress_ratio=1)
    assert _enable(monkeypatch, model_config=_ModelConfig(groups)) is False


@pytest.mark.unit
@pytest.mark.skipif(not ans_utils.AVAILABLE,
                    reason="extension not built with nvcomp")
def test_unsupported_group_dtype_falls_back(monkeypatch, flexkv_log):
    result = _enable(
        monkeypatch,
        model_config=_ModelConfig(_glm53_groups(dtype=torch.float32)))
    assert result is False
    assert "[nvcomp-fallback]" in flexkv_log.text
    assert "does not support" in flexkv_log.text


@pytest.mark.unit
@pytest.mark.skipif(not ans_utils.AVAILABLE,
                    reason="extension not built with nvcomp")
def test_group_dtype_falls_back_to_model_dtype(monkeypatch):
    """``LayerGroupSpec.dtype`` is None until resolved; inherit, don't refuse.

    Both directions are checked, so that reading the model dtype cannot be
    mistaken for ignoring the field.
    """
    groups = _glm53_groups(dtype=None)
    assert _enable(monkeypatch,
                   model_config=_ModelConfig(groups,
                                             dtype=torch.bfloat16)) is True
    assert _enable(monkeypatch,
                   model_config=_ModelConfig(groups,
                                             dtype=torch.float32)) is False


@pytest.mark.unit
@pytest.mark.skipif(not ans_utils.AVAILABLE,
                    reason="extension not built with nvcomp")
def test_compressed_group_with_compress_ratio_falls_back(monkeypatch,
                                                         flexkv_log):
    """A token-compressed leading group would disagree with the table stride."""
    result = _enable(
        monkeypatch,
        model_config=_ModelConfig(_glm53_groups(compress_ratio=64)))
    assert result is False
    assert "compress_ratio" in flexkv_log.text


@pytest.mark.unit
@pytest.mark.skipif(not ans_utils.AVAILABLE,
                    reason="extension not built with nvcomp")
def test_group_layer_count_must_match_cpu_table(monkeypatch, flexkv_log):
    """Group 0 spanning fewer layers than the CPU layout is declined here.

    The strategy raises on the same mismatch, but it runs inside a spawned
    worker where raising hangs the engine on a ``ready`` that never comes --
    the exact failure this guard function exists to avoid. So the declared
    shape has to be rejected in-process first; the worker-side raise stays as
    a backstop against the realized post-TP geometry.
    """
    result = _enable(
        monkeypatch,
        cpu_handle=_Handle(num_layer=78),
        model_config=_ModelConfig(_glm53_groups(num_layers=40)))
    assert result is False
    assert "out of step" in flexkv_log.text


@pytest.mark.unit
@pytest.mark.skipif(not ans_utils.AVAILABLE,
                    reason="extension not built with nvcomp")
def test_small_group_chunk_falls_back(monkeypatch, flexkv_log):
    """The group's own chunk is checked, not just the GPU handles'.

    The handle-level ``min(chunk_sizes)`` guard reads the primary layout,
    which on a multi-group model need not be the group being compressed, so a
    latent group below the ANS floor could otherwise slip past it.
    ``_gpu_groups()`` is left at its healthy 73728 B here precisely so this
    cell can only pass via the per-group check.
    """
    result = _enable(monkeypatch,
                     model_config=_ModelConfig(_glm53_groups(head_size=8)))
    assert result is False
    assert f"{ans_utils.MIN_CHUNK_BYTES}B ANS minimum" in flexkv_log.text


@pytest.mark.unit
@pytest.mark.skipif(not ans_utils.AVAILABLE,
                    reason="extension not built with nvcomp")
def test_ssd_tier_falls_back(monkeypatch, flexkv_log):
    """CPU<->SSD blob-copies a multi-group block and drops the size table.

    The compressed *lengths* live only in DRAM, indexed by CPU block id. A
    DISK2H restore into a different slot would decompress against whatever
    lengths that slot last held -- wrong data, no error. Declined until the
    SSD path can carry the table the way the uniform path does.
    """
    result = _enable(monkeypatch,
                     model_config=_ModelConfig(_glm53_groups()),
                     ssd_handle=_Handle())
    assert result is False
    assert "[nvcomp-fallback]" in flexkv_log.text
    assert "stale length" in flexkv_log.text


@pytest.mark.unit
@pytest.mark.skipif(not ans_utils.AVAILABLE,
                    reason="extension not built with nvcomp")
def test_uniform_model_with_ssd_still_enables(monkeypatch):
    """The SSD refusal is multi-group-only.

    On a uniform model CPU<->SSD packs through ``NvcompCpuSsdStrategy``, which
    carries both size tables, so the hazard above does not exist there.
    """
    assert _enable(monkeypatch, model_config=_ModelConfig(None),
                   ssd_handle=_Handle()) is True
