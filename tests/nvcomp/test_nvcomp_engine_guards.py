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
geometry. Those are *supported*: ``select_nvcomp_groups`` picks whichever
groups the ANS kernels can address, ``FLEXKV_NVCOMP_GROUPS`` narrows that, and
anything not picked moves uncompressed. What has to be pinned down here is that
a well-shaped model is accepted with the right groups selected, that an
unusable group is dropped rather than raised on, and that nvcomp declines
outright only when *nothing* is left to compress.
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


@pytest.fixture(autouse=True)
def _clean_group_selection(monkeypatch):
    """``FLEXKV_NVCOMP_GROUPS`` set in the ambient shell must not steer a cell.

    Every cell here either asserts the default or sets the variable itself, so
    inheriting an operator's value would silently change what is under test --
    and the "auto" cells would fail in a way that looks like a code regression.
    """
    monkeypatch.delenv(ans_utils.NVCOMP_GROUPS_ENV, raising=False)


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
    is page-packed at compress_ratio=64, giving 1 * 1 * 8448 * 1 B = 8448 B --
    both above the ANS floor, so on ``auto`` both are selected.
    """
    latent = dict(num_layers=78, num_kv_heads=1, head_size=576,
                  dtype=torch.bfloat16, compress_ratio=1)
    latent.update(latent_overrides)
    return [
        _Group(**latent),
        _Group(num_layers=78, num_kv_heads=1, head_size=8448,
               dtype=torch.float8_e4m3fn, compress_ratio=64),
    ]


def _plans(groups, *, env="auto", tokens_per_block=64,
           default_dtype=torch.bfloat16, has_ssd=False):
    return ans_utils.select_nvcomp_groups(
        groups, tokens_per_block=tokens_per_block,
        default_dtype=default_dtype, has_ssd=has_ssd, env=env)


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
    # Says which groups it took and how many it left alone, because "nvcomp is
    # on" reads very differently from "nvcomp is on for 90% of the bytes" to
    # someone looking at a ratio in the logs.
    assert "[nvcomp-multi-group]" in flexkv_log.text
    assert "0 other group(s) transfer uncompressed" in flexkv_log.text


@pytest.mark.unit
def test_glm53_auto_selects_both_groups():
    """Both of GLM-5.3's groups clear the ANS floor, so ``auto`` takes both.

    This is the behaviour change: the indexer used to be excluded by fiat.
    Pinned as ordinals *and* table rows, because the row assignment is what
    keeps two compressed groups from overwriting each other's lengths.
    """
    plans = _plans(_glm53_groups())
    assert [p.group_index for p in plans] == [0, 1]
    assert [p.table_layer_offset for p in plans] == [0, 78]
    assert [p.num_layers for p in plans] == [78, 78]


@pytest.mark.unit
@pytest.mark.skipif(not ans_utils.AVAILABLE,
                    reason="extension not built with nvcomp")
def test_accept_is_the_same_fact_as_the_selection(monkeypatch):
    """The engine's accept and the worker's binding must not be two decisions.

    The strategy no longer re-derives which groups to compress; the factory
    resolves ``select_nvcomp_groups`` once and injects the result. This pins
    the other half of that: the engine accepts exactly when that same call
    finds something to do. If the two ever drift, the engine would enable
    nvcomp for a worker that then binds nothing -- or binds a group the engine
    never checked.
    """
    cases = [
        _glm53_groups(),                       # both usable
        _glm53_groups(dtype=torch.float32),    # only group 1
        _glm53_groups(head_size=8),            # only group 1, via the floor
        [_Group(dtype=torch.float32),
         _Group(dtype=torch.float64, head_size=8448)],   # none
    ]
    for groups in cases:
        accepted = _enable(monkeypatch, model_config=_ModelConfig(groups))
        assert accepted is bool(_plans(groups)), groups


@pytest.mark.unit
def test_table_rows_never_overlap():
    """The invariant the whole multi-group design rests on.

    Two groups sharing a row would silently decompress at each other's
    lengths. Checked over every group, not just the selected ones, because the
    table is sized and offset for all of them so a later selection change does
    not need a reallocation.
    """
    groups = _glm53_groups()
    groups.append(_Group(num_layers=12, head_size=576))
    offsets = ans_utils.group_table_layer_offsets(groups)
    spans = [range(o, o + g.num_layers) for o, g in zip(offsets, groups)]
    seen = set()
    for span in spans:
        assert not (seen & set(span)), "layer-group table rows overlap"
        seen |= set(span)
    assert max(seen) + 1 == sum(g.num_layers for g in groups)


@pytest.mark.unit
def test_env_can_narrow_to_one_group():
    """``FLEXKV_NVCOMP_GROUPS=0`` restores the old single-group behaviour."""
    plans = _plans(_glm53_groups(), env="0")
    assert [p.group_index for p in plans] == [0]
    assert plans[0].table_layer_offset == 0


@pytest.mark.unit
def test_env_can_name_several_groups():
    plans = _plans(_glm53_groups(), env="0,1")
    assert [p.group_index for p in plans] == [0, 1]


@pytest.mark.unit
@pytest.mark.skipif(not ans_utils.AVAILABLE,
                    reason="extension not built with nvcomp")
def test_env_none_disables(monkeypatch, flexkv_log):
    """``none`` is a kill switch, so it has to reach the engine's decision.

    Asserted through ``check_engine_nvcomp_enable`` rather than
    ``select_nvcomp_groups`` because that is the call whose False keeps a
    worker from ever binding a strategy.
    """
    monkeypatch.setenv(ans_utils.NVCOMP_GROUPS_ENV, "none")
    assert _enable(monkeypatch,
                   model_config=_ModelConfig(_glm53_groups())) is False
    assert "selects no layer groups" in flexkv_log.text


@pytest.mark.unit
def test_env_is_read_when_not_passed(monkeypatch):
    """The env is the live knob; ``env=`` exists only so tests need not set it."""
    monkeypatch.setenv(ans_utils.NVCOMP_GROUPS_ENV, "1")
    plans = ans_utils.select_nvcomp_groups(
        _glm53_groups(), tokens_per_block=64, default_dtype=torch.bfloat16)
    assert [p.group_index for p in plans] == [1]
    assert plans[0].table_layer_offset == 78


@pytest.mark.unit
def test_env_naming_an_ineligible_group_declines(flexkv_log):
    """Asked for a group ANS cannot take -> decline, do not silently narrow.

    Quietly dropping it would leave an operator believing a measurement
    covered a group it never touched, which is the failure mode this whole
    selection path exists to make impossible.
    """
    groups = _glm53_groups(dtype=torch.float32)
    assert _plans(groups, env="0") == []
    assert "names group(s) ANS cannot compress" in flexkv_log.text


@pytest.mark.unit
def test_env_out_of_range_declines(flexkv_log):
    assert _plans(_glm53_groups(), env="5") == []
    assert "out of range" in flexkv_log.text


@pytest.mark.unit
def test_env_garbage_declines(flexkv_log):
    assert _plans(_glm53_groups(), env="latent") == []
    assert "not an integer" in flexkv_log.text


@pytest.mark.unit
@pytest.mark.skipif(not ans_utils.AVAILABLE,
                    reason="extension not built with nvcomp")
def test_three_groups_are_accepted(monkeypatch, flexkv_log):
    """Two groups is GLM-5.3's shape, not the supported limit.

    DSv4 registers more than two, and all three here are addressable, so all
    three are taken.
    """
    groups = _glm53_groups()
    groups.append(_Group(head_size=8448, dtype=torch.float8_e4m3fn,
                         compress_ratio=4))
    assert _enable(monkeypatch, model_config=_ModelConfig(groups)) is True
    assert "0 other group(s) transfer uncompressed" in flexkv_log.text
    assert [p.group_index for p in _plans(groups)] == [0, 1, 2]


@pytest.mark.unit
@pytest.mark.skipif(not ans_utils.AVAILABLE,
                    reason="extension not built with nvcomp")
def test_ineligible_group_is_dropped_not_fatal(monkeypatch, flexkv_log):
    """One unusable group no longer costs the whole model its compression.

    Under the old single-group rule a broken group 0 declined nvcomp outright
    even when group 1 was perfectly compressible. Now the usable group is
    taken and the other is reported as skipped -- which is the *only* reason
    it is safe for the strategy to bind more than one group.
    """
    groups = _glm53_groups(dtype=torch.float32)  # group 0 unusable
    assert _enable(monkeypatch, model_config=_ModelConfig(groups)) is True
    assert [p.group_index for p in _plans(groups)] == [1]
    assert "transfers uncompressed" in flexkv_log.text
    assert "1 other group(s) transfer uncompressed" in flexkv_log.text


@pytest.mark.unit
@pytest.mark.skipif(not ans_utils.AVAILABLE,
                    reason="extension not built with nvcomp")
def test_all_groups_unsupported_falls_back(monkeypatch, flexkv_log):
    """Nothing left to compress is the one case that still declines."""
    groups = [_Group(dtype=torch.float32),
              _Group(dtype=torch.float64, head_size=8448)]
    assert _enable(monkeypatch, model_config=_ModelConfig(groups)) is False
    assert "no layer group can be compressed" in flexkv_log.text


@pytest.mark.unit
def test_group_dtype_falls_back_to_model_dtype():
    """``LayerGroupSpec.dtype`` is None until resolved; inherit, don't refuse.

    Both directions are checked, so that reading the model dtype cannot be
    mistaken for ignoring the field. Group 1 keeps its own fp8, so the
    unsupported case is asserted on group 0's selection rather than on the
    model being declined outright.
    """
    groups = _glm53_groups(dtype=None)
    assert 0 in [p.group_index for p in
                 _plans(groups, default_dtype=torch.bfloat16)]
    assert 0 not in [p.group_index for p in
                     _plans(groups, default_dtype=torch.float32)]


@pytest.mark.unit
def test_compress_ratio_no_longer_excludes_a_group():
    """A token-compressed group is compressible now that rows are per-group.

    It used to be refused because its chunk disagreed with a table stride
    shared across groups. With each group owning its own rows, the only thing
    compress_ratio changes is the chunk size -- which is checked on its own.
    """
    plans = _plans(_glm53_groups(compress_ratio=2, head_size=576))
    assert 0 in [p.group_index for p in plans]


@pytest.mark.unit
def test_group_layer_count_no_longer_has_to_match_cpu_table():
    """A group narrower than the PP stage gets its own rows, not a refusal.

    The old rule required group 0 to span every layer, because the table had
    one layer dimension sized from the CPU layout. The table is now sized from
    the summed group layer counts, so a 40-layer group simply occupies 40 rows.
    """
    groups = _glm53_groups(num_layers=40)
    plans = _plans(groups)
    assert [(p.group_index, p.table_layer_offset, p.num_layers)
            for p in plans] == [(0, 0, 40), (1, 40, 78)]


@pytest.mark.unit
def test_small_group_chunk_is_dropped(flexkv_log):
    """The group's own chunk is checked, not just the GPU handles'.

    The handle-level ``min(chunk_sizes)`` guard reads the primary layout,
    which on a multi-group model need not be a group being compressed, so a
    latent group below the ANS floor could otherwise slip past it.
    ``_gpu_groups()`` is left at its healthy 73728 B here precisely so this
    cell can only pass via the per-group check.
    """
    groups = _glm53_groups(head_size=8)  # 64 * 1 * 8 * 2 = 1024 B
    assert [p.group_index for p in _plans(groups)] == [1]
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
