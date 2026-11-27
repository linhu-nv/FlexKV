"""FlexKV's vLLM connector on the scheduler side, KVManager faked: the stats
it reports, the group a load restores, its prefetches, when vLLM waits for
its PUTs to free blocks, and how a failed load reaches vLLM."""
import importlib.util
import time
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

pytest.importorskip("vllm.distributed.kv_transfer.kv_connector.v1.base")

from vllm.v1.engine import FinishReason  # noqa: E402
from vllm.v1.outputs import KVConnectorOutput  # noqa: E402

from flexkv.common.config import RankInfo  # noqa: E402
from flexkv.common.request import KVResponseStatus  # noqa: E402
from flexkv.integration.config import FlexKVConfig, VllmKVGroup, VllmSWAGroups  # noqa: E402
from flexkv.integration.stats import FlexKVStats  # noqa: E402
from flexkv.integration.vllm import vllm_v1_adapter as adapter  # noqa: E402

pytestmark = pytest.mark.unit

needs_kv_metrics = pytest.mark.skipif(
    importlib.util.find_spec("vllm.distributed.kv_transfer.kv_connector.v1.metrics") is None,
    reason="vLLM without KV connector metrics")
needs_transfer_results = pytest.mark.skipif(
    adapter.KVConnectorTransferResults is None,
    reason="vLLM without KVConnectorTransferResults")

Impl = adapter.FlexKVConnectorV1Impl
REQ = "req-0"
TASK = 7
PREFETCH = 5


@pytest.fixture
def manager(monkeypatch):
    manager = MagicMock()
    manager.is_ready.return_value = True
    manager.has_running_graphs.return_value = False
    manager.prefetch_async.return_value = PREFETCH
    manager.try_wait.return_value = {}
    manager.get_match.return_value = (9, np.zeros(32, dtype=bool))
    monkeypatch.setattr(adapter, "KVManager", MagicMock(return_value=manager))
    return manager


def _impl(role, connector=None):
    impl = object.__new__(Impl)
    impl.role = role
    impl.connector = connector
    impl.previous_scheduled_req_ids = set()
    impl._bound_metadata = None
    return impl


def _scheduler(config=None):
    config = config or FlexKVConfig()
    return _impl(adapter.KVConnectorRole.SCHEDULER, adapter.FlexKVSchedulerConnector(
        config, RankInfo(model_config=config.model_config), enable_layerwise_transfer=False))


@pytest.fixture
def scheduler(manager):
    return _scheduler()


@pytest.fixture
def worker():
    return _impl(adapter.KVConnectorRole.WORKER)


@needs_kv_metrics
def test_take_counts_hands_each_count_out_once_across_log_windows():
    stats = FlexKVStats(2)
    stats.record_get(num_prompt_tokens=64, num_gpu_matched_tokens=16,
                     num_flexkv_matched_tokens=32)
    # The second GET closes FlexKV's own log window, which clears its counters.
    stats.record_get(num_prompt_tokens=64, num_gpu_matched_tokens=0,
                     num_flexkv_matched_tokens=48)
    stats.record_put(num_all_tokens=64, num_unmatched_tokens=16)

    assert stats.num_get_requests == 0
    assert stats.take_counts() == {
        "num_get_requests": 2, "num_get_query_tokens": 128,
        "num_gpu_matched_tokens": 16, "num_flexkv_matched_tokens": 80,
        "num_put_requests": 1, "num_put_query_tokens": 64,
        "num_put_unmatched_tokens": 16,
    }
    assert stats.take_counts() == {}


@needs_kv_metrics
def test_scheduler_reports_the_counts_of_each_step():
    impl = _impl(adapter.KVConnectorRole.SCHEDULER,
                 SimpleNamespace(flexkv_stats=FlexKVStats(200)))
    impl.connector.flexkv_stats.record_get(
        num_prompt_tokens=64, num_gpu_matched_tokens=0, num_flexkv_matched_tokens=64)

    assert impl.get_kv_connector_stats().data["num_get_requests"] == 1
    assert impl.get_kv_connector_stats().is_empty()


@needs_kv_metrics
def test_logged_stats_sum_the_step_reports():
    # What vLLM's KVConnectorLogging does with the payloads it receives.
    total = Impl.build_kv_connector_stats(
        {"num_get_query_tokens": 64, "num_flexkv_matched_tokens": 64})
    total.aggregate(Impl.build_kv_connector_stats(
        {"num_get_query_tokens": 64, "num_gpu_matched_tokens": 32}))

    metrics = total.reduce()
    assert metrics["num_get_query_tokens"] == 128
    assert metrics["get_flexkv_match_ratio"] == 0.5
    assert metrics["get_gpu_match_ratio"] == 0.25
    assert Impl.build_kv_connector_stats(None).is_empty()


def test_connector_hooks_ask_for_what_flexkv_needs():
    assert not object.__new__(Impl).requires_kv_delivery
    assert not Impl.supports_lookahead_block_hashes(None)
    assert Impl.get_required_kvcache_layout(None) == "LBNHC"
    assert Impl.requires_piecewise_for_cudagraph({"use_layerwise": True})
    assert not Impl.requires_piecewise_for_cudagraph({})


def test_the_extensible_kv_cache_stays_off():
    access = getattr(pytest.importorskip("vllm.config.kv_transfer"), "KVCacheAccess", None)
    if access is None:
        pytest.skip("vLLM without KVCacheAccess")
    assert Impl.kv_cache_access(None) == access.LEGACY_IPC


@pytest.mark.parametrize("group, loaded", [
    pytest.param(None, (0,), id="one-group"),
    pytest.param(VllmKVGroup(1, ("layers.0.attn",)), (1,), id="hybrid"),
])
def test_a_load_restores_the_group_flexkv_stores(manager, group, loaded):
    config = FlexKVConfig()
    config.vllm_kv_group = group

    assert _scheduler(config).get_loaded_kv_cache_group_ids(MagicMock()) == loaded


@pytest.fixture
def prefetching(scheduler):
    scheduler.connector.enable_prefetch = True
    return scheduler


def _arriving(num_tokens=40):
    return SimpleNamespace(request_id=REQ, num_tokens=num_tokens,
                           all_token_ids=list(range(num_tokens)))


def test_prefetch_starts_on_arrival_and_gates_the_match(prefetching, manager):
    request = _arriving()
    prefetching.on_new_request(request)

    manager.prefetch_async.assert_called_once()
    assert list(manager.prefetch_async.call_args.kwargs["token_ids"]) == list(range(32))

    connector = prefetching.connector
    assert connector.get_num_new_matched_tokens(request, 16) == (None, False)
    manager.get_match.assert_not_called()

    manager.try_wait.return_value = {PREFETCH: SimpleNamespace()}
    assert connector.get_num_new_matched_tokens(request, 16) == (0, False)
    manager.prefetch_async.assert_called_once()
    manager.get_match.assert_called_once()


def test_request_without_a_full_block_has_nothing_to_wait_for(prefetching, manager):
    request = _arriving(num_tokens=10)
    prefetching.on_new_request(request)

    manager.prefetch_async.assert_not_called()
    assert prefetching.connector.get_num_new_matched_tokens(request, 0) == (0, False)


def test_scheduling_does_not_start_a_missing_prefetch(prefetching, manager):
    # A vLLM that does not call on_new_request gets no prefetch.
    assert prefetching.connector.get_num_new_matched_tokens(_arriving(), 0) == (0, False)

    manager.prefetch_async.assert_not_called()
    manager.get_match.assert_called_once()


def test_request_ended_before_scheduling_cancels_its_prefetch(prefetching, manager):
    request = _arriving()
    request.is_finished = lambda: True
    request.get_finished_reason = lambda: FinishReason.ABORT
    prefetching.on_new_request(request)

    prefetching.connector.request_finished(request, [])

    manager.cancel.assert_called_once_with(task_ids=[PREFETCH])
    assert not prefetching.connector.prefetch_tasks


def test_no_prefetch_without_a_tier_to_prefetch_from(scheduler, manager):
    scheduler.connector.enable_prefetch = False
    scheduler.on_new_request(_arriving())

    manager.prefetch_async.assert_not_called()


@pytest.fixture
def waiting(scheduler):
    scheduler.connector.pending_free_wait_s = 1.0
    return scheduler


def _task(cls, task_id, launched_ago=None, **fields):
    task = cls(task_id=task_id, request=SimpleNamespace(request_id=f"req-{task_id}"), **fields)
    if launched_ago is not None:
        task.task_launch_time = time.perf_counter() - launched_ago
    return task


def _put(task_id, launched_ago=None):
    return _task(adapter.FlexKVPutTask, task_id, launched_ago,
                 num_matched_tokens=0, num_unmatched_tokens=16)


def test_nothing_to_wait_for_without_a_put(waiting):
    waiting.connector.tasks_to_launch[1] = _task(
        adapter.FlexKVGetTask, 1, num_computed_tokens=0, num_new_matched_tokens=16)

    assert not waiting.has_pending_block_frees()


def test_a_put_about_to_launch_frees_blocks_soon(waiting):
    waiting.connector.tasks_to_launch[1] = _put(1)

    assert waiting.has_pending_block_frees()


def test_vllm_waits_for_puts_until_the_oldest_is_overdue(waiting):
    connector = waiting.connector
    connector.put_tasks[1] = [_put(1, launched_ago=0.5)]
    connector.put_tasks[2] = [_put(2, launched_ago=0.0)]
    assert waiting.has_pending_block_frees()

    connector.put_tasks[1][0].task_launch_time -= 1.0
    assert not waiting.has_pending_block_frees()


def test_a_zero_wait_never_waits(waiting):
    waiting.connector.pending_free_wait_s = 0
    waiting.connector.tasks_to_launch[1] = _put(1)

    assert not waiting.has_pending_block_frees()


def test_the_worker_side_never_waits(worker):
    assert not worker.has_pending_block_frees()


def _finished(reason=FinishReason.ABORT, offload=False):
    return SimpleNamespace(
        request_id=REQ, num_tokens=33, is_finished=lambda: True,
        get_finished_reason=lambda: reason, offload_kv_on_finish=offload)


def _finish_get(scheduler, manager, status):
    """End an async GET of REQ with ``status`` and let vLLM poll it."""
    request = _finished()
    connector = scheduler.connector
    connector.req_id_to_task_dict[REQ] = TASK
    connector.get_tasks[TASK] = [adapter.FlexKVGetTask(
        task_id=TASK, request=request, num_computed_tokens=0, num_new_matched_tokens=16)]
    manager.try_wait.return_value = {TASK: SimpleNamespace(status=status)}
    output = KVConnectorOutput()
    scheduler.update_connector_output(output)
    manager.try_wait.return_value = {}
    return request, output


def _step(scheduler):
    return scheduler.build_connector_meta(
        SimpleNamespace(preempted_req_ids=set(), new_block_ids_to_zero=None))


@needs_transfer_results
def test_failed_get_is_reported_by_the_workers_next_step(scheduler, worker, manager):
    request, output = _finish_get(scheduler, manager, KVResponseStatus.FAILED)
    assert not output.finished_recving

    metadata = _step(scheduler)
    assert metadata.failed_recving == {REQ}
    assert not _step(scheduler).failed_recving

    worker.bind_connector_metadata(metadata)
    results = worker.get_transfer_results(set())
    assert results.finished_recving == results.failed_recving == {REQ}
    assert worker.get_finished(set()) == (None, {REQ})

    output = KVConnectorOutput(finished_recving={REQ}, failed_recving={REQ})
    scheduler.update_connector_output(output)
    assert output.finished_recving == {REQ}

    # vLLM recomputes the request: it is not loaded again.
    assert scheduler.connector.get_num_new_matched_tokens(request, 0) == (0, False)
    manager.get_match.assert_not_called()
    scheduler.connector.request_finished(request, [0, 1])
    assert REQ not in scheduler.connector.failed_load_req_ids


@needs_transfer_results
def test_request_ended_before_its_failed_load_is_reported_stores_nothing(scheduler, manager):
    _finish_get(scheduler, manager, KVResponseStatus.FAILED)

    delay = scheduler.connector.request_finished(_finished(offload=True), [0, 1])

    assert delay is False
    manager.put_match.assert_not_called()
    # vLLM still waits for the report to free the blocks.
    assert _step(scheduler).failed_recving == {REQ}


@needs_transfer_results
def test_worker_reads_metadata_bound_on_the_vllm_wrapper(worker, monkeypatch):
    import vllm.distributed.kv_transfer as kv_transfer

    metadata = adapter.FlexKVConnectorMetadata(failed_recving=frozenset({REQ}))
    outer = SimpleNamespace(has_connector_metadata=lambda: True,
                            _get_connector_metadata=lambda: metadata)
    monkeypatch.setattr(kv_transfer, "get_kv_transfer_group", lambda: outer)

    assert worker.get_finished(set()) == (None, {REQ})


@needs_transfer_results
def test_successful_get_is_reported_finished_at_once(scheduler, manager):
    _, output = _finish_get(scheduler, manager, KVResponseStatus.SUCCESS)

    assert output.finished_recving == {REQ}
    assert not _step(scheduler).failed_recving


@needs_transfer_results
def test_vllm_without_failed_receives_takes_a_failed_get_as_finished(scheduler, manager):
    scheduler.connector.report_failed_loads = False

    _, output = _finish_get(scheduler, manager, KVResponseStatus.FAILED)

    assert output.finished_recving == {REQ}
    assert not _step(scheduler).failed_recving


SWA = VllmSWAGroups(group_ids=(0, 1), block_size=32, window_blocks=4,
                    block_bytes=1024, region_layer_names=("layers.0.attn.swa_cache",))
SNAPSHOT = 11
# A 200-token prompt: full-attention group 2 in 64-token blocks, the two
# sliding-window groups in 32-token blocks; it snapshots at token 192.
BLOCK_IDS = ([20, 21, 22, 23, 24, 25, 26], [30, 31, 32, 33, 34, 35, 36], [10, 11, 12, 13])
WINDOW = [22, 23, 24, 25, 32, 33, 34, 35]


class _BlockPool:
    """Stands in for vLLM's BlockPool: counts the references FlexKV takes."""

    def __init__(self):
        self.blocks = [SimpleNamespace(block_id=i, ref_cnt=1) for i in range(40)]
        self.null_block = self.blocks[0]

    def touch(self, blocks):
        for block in blocks:
            block.ref_cnt += 1

    def free_blocks(self, blocks):
        for block in blocks:
            block.ref_cnt -= 1

    def refs(self, block_ids):
        return [self.blocks[i].ref_cnt for i in block_ids]


@pytest.fixture
def swa(manager):
    config = FlexKVConfig()
    config.cache_config.tokens_per_block = 64
    config.vllm_kv_group = VllmKVGroup(2, ("layers.0.attn",), swa=SWA)
    scheduler = _scheduler(config)
    pool = _BlockPool()
    scheduler.bind_kv_cache_manager(SimpleNamespace(block_pool=pool))
    manager.put_match.return_value = (SNAPSHOT, np.ones(192, dtype=bool))
    manager.launch.side_effect = lambda task_ids, **kwargs: list(task_ids)
    return scheduler, pool


def _prompt(finished_reason=None, request_id=REQ):
    request = SimpleNamespace(
        request_id=request_id, num_tokens=200, num_prompt_tokens=200,
        all_token_ids=list(range(200)), num_computed_tokens=0,
        num_in_flight_tokens=0, num_stale_output_tokens=0)
    request.is_finished = lambda: finished_reason is not None
    request.get_finished_reason = lambda: finished_reason
    return request


def _prompt_step(num_computed, num_scheduled, preempted=(), req_ids=(REQ,), block_ids=BLOCK_IDS):
    return SimpleNamespace(
        scheduled_new_reqs=[SimpleNamespace(req_id=req_id, num_computed_tokens=num_computed)
                            for req_id in req_ids],
        scheduled_cached_reqs=SimpleNamespace(req_ids=[], num_computed_tokens=[]),
        num_scheduled_tokens=dict.fromkeys(req_ids, num_scheduled),
        kv_connector_block_state=SimpleNamespace(get_block_ids=lambda req_id: block_ids),
        preempted_req_ids=set(preempted), finished_req_ids=set())


def _launches(manager):
    return [c for c in manager.launch.call_args_list if c.kwargs["task_ids"] == [SNAPSHOT]]


def test_prompt_end_snapshot_pins_its_blocks_until_it_lands(swa, manager):
    scheduler, pool = swa
    request = _prompt()
    scheduler.on_new_request(request)

    scheduler.build_connector_meta(_prompt_step(0, 128))
    assert not scheduler.connector.pending_snapshots

    scheduler.build_connector_meta(_prompt_step(128, 72))
    pinned = [10, 11, 12] + WINDOW
    assert pool.refs(pinned) == [2] * len(pinned)

    request.num_computed_tokens, request.num_in_flight_tokens = 200, 72
    scheduler.update_connector_output(KVConnectorOutput())
    manager.put_match.assert_not_called()

    request.num_in_flight_tokens = 0
    scheduler.update_connector_output(KVConnectorOutput())
    kwargs = manager.put_match.call_args.kwargs
    assert len(kwargs["token_ids"]) == 192 and kwargs["swa"] is True
    (launch,) = _launches(manager)
    tpb = 64
    assert list(launch.kwargs["slot_mappings"][0][::tpb] // tpb) == [10, 11, 12]
    assert list(launch.kwargs["swa_slot_mappings"][0][::tpb] // tpb) == WINDOW

    manager.try_wait.return_value = {SNAPSHOT: SimpleNamespace(status=KVResponseStatus.SUCCESS)}
    output = KVConnectorOutput()
    scheduler.update_connector_output(output)
    assert pool.refs(pinned) == [1] * len(pinned)
    assert not output.finished_sending


def test_a_short_prompt_takes_no_snapshot(swa, manager):
    scheduler, _ = swa
    request = _prompt()
    request.num_prompt_tokens = 128
    scheduler.on_new_request(request)

    scheduler.build_connector_meta(_prompt_step(0, 128))

    assert not scheduler.connector.pending_snapshots


@pytest.mark.parametrize("reason, num_in_flight, launched", [
    pytest.param(FinishReason.STOP, 0, True, id="stop"),
    pytest.param(FinishReason.STOP, 200, False, id="stop-in-flight"),
    pytest.param(FinishReason.ABORT, 0, False, id="abort"),
])
def test_finishing_settles_a_pending_snapshot(swa, manager, reason, num_in_flight, launched):
    scheduler, pool = swa
    request = _prompt(reason)
    scheduler.on_new_request(request)
    scheduler.build_connector_meta(_prompt_step(0, 200))
    request.num_computed_tokens, request.num_in_flight_tokens = 200, num_in_flight

    scheduler.request_finished(request, [10, 11, 12, 13])
    scheduler.update_connector_output(KVConnectorOutput())

    assert bool(_launches(manager)) == launched
    assert pool.refs(WINDOW) == [2 if launched else 1] * len(WINDOW)
    # The snapshot stores what the request's own Full-only PUT would.
    assert [c.kwargs["swa"] for c in manager.put_match.call_args_list] == (
        [True] if launched else [False] if reason == FinishReason.STOP else [])


def test_snapshots_ready_in_one_step_launch_as_one_batch(swa, manager):
    scheduler, pool = swa
    requests = [_prompt(request_id=f"req-{i}") for i in range(2)]
    for request in requests:
        scheduler.on_new_request(request)
    manager.put_match.side_effect = [(SNAPSHOT, np.ones(192, dtype=bool)),
                                     (SNAPSHOT + 1, np.ones(192, dtype=bool))]
    manager.launch.side_effect = lambda task_ids, **kwargs: [TASK]
    scheduler.build_connector_meta(_prompt_step(0, 200, req_ids=("req-0", "req-1")))
    assert pool.refs(WINDOW) == [3] * len(WINDOW)

    for request in requests:
        request.num_computed_tokens = 200
    scheduler.update_connector_output(KVConnectorOutput())
    launch = manager.launch.call_args.kwargs
    assert launch["task_ids"] == [SNAPSHOT, SNAPSHOT + 1] and launch["as_batch"]

    manager.try_wait.return_value = {TASK: SimpleNamespace(status=KVResponseStatus.SUCCESS)}
    scheduler.update_connector_output(KVConnectorOutput())
    assert pool.refs(WINDOW) == [1] * len(WINDOW)


def test_preemption_drops_a_pending_snapshot(swa, manager):
    scheduler, pool = swa
    scheduler.on_new_request(_prompt())
    scheduler.build_connector_meta(_prompt_step(0, 200))

    scheduler.build_connector_meta(_prompt_step(0, 0, preempted=[REQ]))
    scheduler.update_connector_output(KVConnectorOutput())

    manager.put_match.assert_not_called()
    assert pool.refs(WINDOW) == [1] * len(WINDOW)
    assert not _launches(manager)


def test_a_prefill_chunk_snapshots_the_last_interval_multiple_it_reaches(swa, manager):
    scheduler, pool = swa
    scheduler.connector.swa = replace(SWA, interval=128)
    request = _prompt()
    scheduler.on_new_request(request)
    window = [20, 21, 22, 23, 30, 31, 32, 33]

    scheduler.build_connector_meta(_prompt_step(0, 160))
    assert pool.refs([10, 11] + window) == [2] * 10

    manager.put_match.return_value = (SNAPSHOT, np.ones(128, dtype=bool))
    request.num_computed_tokens = 160
    scheduler.update_connector_output(KVConnectorOutput())
    assert len(manager.put_match.call_args.kwargs["token_ids"]) == 128
    (launch,) = _launches(manager)
    assert list(launch.kwargs["slot_mappings"][0][::64] // 64) == [10, 11]
    assert list(launch.kwargs["swa_slot_mappings"][0][::64] // 64) == window


def test_decode_snapshots_every_interval(swa, manager):
    scheduler, pool = swa
    scheduler.connector.swa = replace(SWA, interval=256)
    request = _prompt()
    request.all_token_ids = list(range(256))
    scheduler.on_new_request(request)
    # Token 255 in decode; the sliding windows freed their first blocks.
    block_ids = ([0, 0, 22, 23, 24, 25, 26, 27], [0, 0, 32, 33, 34, 35, 36, 37], [10, 11, 12, 13])
    window = [24, 25, 26, 27, 34, 35, 36, 37]

    scheduler.build_connector_meta(_prompt_step(254, 1, block_ids=block_ids))
    assert not scheduler.connector.pending_snapshots
    scheduler.build_connector_meta(_prompt_step(255, 1, block_ids=block_ids))
    assert pool.refs([10, 11, 12, 13] + window) == [2] * 12

    manager.put_match.return_value = (SNAPSHOT, np.arange(256) >= 192)
    request.num_computed_tokens, request.num_in_flight_tokens = 256, 1
    scheduler.update_connector_output(KVConnectorOutput())
    manager.put_match.assert_not_called()

    request.num_in_flight_tokens = 0
    scheduler.update_connector_output(KVConnectorOutput())
    assert len(manager.put_match.call_args.kwargs["token_ids"]) == 256
    (launch,) = _launches(manager)
    tpb = 64
    assert list(launch.kwargs["slot_mappings"][0][::tpb] // tpb) == [13]
    assert list(launch.kwargs["swa_slot_mappings"][0][::tpb] // tpb) == window


@pytest.mark.parametrize("interval", [0, 1024])
@pytest.mark.parametrize("reason", [FinishReason.STOP, FinishReason.ABORT])
def test_a_request_ending_in_decode_snapshots_its_last_window(swa, manager, reason, interval):
    scheduler, pool = swa
    scheduler.connector.swa = replace(SWA, interval=interval)
    request = _prompt(reason)
    request.num_tokens, request.all_token_ids = 321, list(range(321))
    scheduler.on_new_request(request)
    block_ids = ([0, 0, 0, 0, 24, 25, 26, 27, 28, 29], [0, 0, 0, 0, 34, 35, 36, 37, 38, 39],
                 [10, 11, 12, 13, 14])
    window = [26, 27, 28, 29, 36, 37, 38, 39]
    # Windows at 256 and 320, neither a multiple of the interval: pinned, not stored.
    for num_computed in (255, 319):
        request.num_computed_tokens = num_computed
        scheduler.build_connector_meta(_prompt_step(num_computed, 1, block_ids=block_ids))
        request.num_computed_tokens = num_computed + 1
        scheduler.update_connector_output(KVConnectorOutput())
    assert not _launches(manager)
    assert pool.refs([24, 26, 28]) == [2, 3, 2]

    manager.put_match.return_value = (SNAPSHOT, np.arange(320) >= 256)
    scheduler.request_finished(request, [10, 11, 12, 13, 14])
    scheduler.update_connector_output(KVConnectorOutput())

    assert pool.refs([24, 25, 34, 35]) == [1] * 4
    if reason == FinishReason.STOP:
        (launch,) = _launches(manager)
        assert list(launch.kwargs["slot_mappings"][0][::64] // 64) == [14]
        assert list(launch.kwargs["swa_slot_mappings"][0][::64] // 64) == window
        assert pool.refs(window) == [2] * len(window)
        # The snapshot at token 320 stores what the request's own PUT would.
        assert [c.kwargs["swa"] for c in manager.put_match.call_args_list] == [True]
    else:
        assert not _launches(manager)
        assert pool.refs(window) == [1] * len(window)


def test_a_hit_loads_the_window_before_its_end(swa, manager):
    scheduler, _ = swa
    request = _prompt()
    manager.get_match.return_value = (TASK, np.ones(192, dtype=bool))

    assert scheduler.get_num_new_matched_tokens(request, 0) == (192, True)
    assert manager.get_match.call_args.kwargs["swa_aware"] is True

    blocks = SimpleNamespace(get_block_ids=lambda: BLOCK_IDS)
    scheduler.update_state_after_alloc(request, blocks, 192)
    task = scheduler.connector.tasks_to_launch[TASK]
    assert list(task.swa_slot_mapping[::64] // 64) == WINDOW
    assert scheduler.get_loaded_kv_cache_group_ids(request) == (2, 0, 1)


def test_a_layerwise_hit_keeps_its_window_from_zeroing(swa, manager):
    scheduler, _ = swa
    request = _prompt()
    manager.get_match.return_value = (TASK, np.ones(192, dtype=bool))
    scheduler.get_num_new_matched_tokens(request, 0)
    scheduler.update_state_after_alloc(request, SimpleNamespace(get_block_ids=lambda: BLOCK_IDS), 192)
    scheduler.connector.enable_layerwise_transfer = True
    step = SimpleNamespace(new_block_ids_to_zero=[10, 12, 13, 22, 26, 35, 36])

    scheduler.connector.keep_loads_from_zeroing(step)

    assert step.new_block_ids_to_zero == [13, 26, 36]


def test_a_hit_shorter_than_the_window_is_dropped(swa, manager):
    scheduler, _ = swa
    manager.get_match.return_value = (TASK, np.ones(64, dtype=bool))

    assert scheduler.get_num_new_matched_tokens(_prompt(), 0) == (0, False)
    manager.cancel.assert_called_with(task_ids=[TASK])
