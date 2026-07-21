"""Opt-in two-rank radixshmem/FlexKV peer-match integration test."""
from __future__ import annotations

import multiprocessing as mp
import os
import time
import traceback

import numpy as np
import pytest


class _PeerDirectory:
    def resolve_radix_rank(self, cluster_id, rank):
        return 100 + int(rank)


def _rank_main(rank, prefix, port, rdma_dev, ready, done, output):
    try:
        import shmradix

        from flexkv.cache.radix_shmem_engine import CacheEngineRadixShmem
        from flexkv.cache.get_planner import get_media_list, route_of
        from flexkv.common.transfer import DeviceType
        from flexkv.common.type import MatchResult, MatchResultAccel

        shm = shmradix.ShmConfig(
            max_nodes=1170,
            max_blocks=1170,
            block_size=16,
            data_pool_ratio=8,
            background_evict=True,
        )
        cfg = shmradix.RadixServerConfig()
        cfg.name = prefix
        cfg.shm = shm
        cfg.rank = rank
        cfg.world_size = 2
        cfg.master_addr = "127.0.0.1"
        cfg.master_port = port
        cfg.rdma_dev = rdma_dev
        cfg.gid_idx = int(os.getenv("FLEXKV_RADIX_GID_IDX", "3"))
        cfg.bootstrap_timeout_sec = 30

        server = shmradix.RadixServer(cfg)
        if not server.bootstrap():
            raise RuntimeError("server bootstrap returned false")

        hashes = np.arange(26, dtype=np.uint64) * 104729 + 101
        query_hashes = hashes[:-1]
        if rank == 0:
            engine = CacheEngineRadixShmem(
                device_type=DeviceType.CPU,
                num_total_blocks=1170,
                tokens_per_block=16,
                shm_name=server.shm_name(),
                peer_enabled=True,
                redis_meta=_PeerDirectory(),
                radix_cluster_id="pytest",
            )
            sequence = type("Seq", (), {
                "block_hashes": hashes.view(np.int64),
                "gen_hashes": lambda self: None,
            })()
            slots = engine.take(num_required_blocks=len(hashes), strict=True)
            if len(slots) != len(hashes):
                raise RuntimeError(f"took={len(slots)}")
            node, _unused = engine.insert(
                sequence, slots,
                num_insert_blocks=len(hashes),
                is_ready=False,
            )
            engine.unlock(node)
            ready.set()
            if not done.wait(20):
                raise TimeoutError("reader did not complete")
        else:
            engine = CacheEngineRadixShmem(
                device_type=DeviceType.CPU,
                num_total_blocks=2048,
                tokens_per_block=16,
                shm_name=server.shm_name(),
                peer_enabled=True,
                redis_meta=_PeerDirectory(),
                radix_cluster_id="pytest",
            )
            if not ready.wait(20):
                raise TimeoutError("writer did not publish")
            result = None
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                result = engine.match(
                    type("Seq", (), {
                        "block_hashes": query_hashes.view(np.int64),
                        "gen_hashes": lambda self: None,
                    })()
                )
                if result.peer_ready == len(query_hashes):
                    break
                guard = (
                    result.remote.pre_locked_node
                    if result.remote is not None
                    else result.local.pre_locked_node
                )
                if guard is not None:
                    engine.unlock(guard)
                time.sleep(0.01)
            if result is None or result.peer_ready != len(query_hashes):
                raise AssertionError("remote prefix was not visible")
            peer = result.remote
            assert peer is not None
            empty_ssd = MatchResult(local=MatchResultAccel())
            # The media list is config-agnostic; only the per-segment route
            # depends on enable_peer_gpu (staged PEERH2H vs direct PEERH2D).
            segments = get_media_list(
                result, empty_ssd,
                block_mask_start=0,
                block_mask_end=len(query_hashes),
            )
            staged_type = route_of(
                segments[-1], enable_gpu=True, enable_gds=False,
                enable_peer_gpu=False,
            ).primary_type.value
            direct_type = route_of(
                segments[-1], enable_gpu=True, enable_gds=False,
                enable_peer_gpu=True,
            ).primary_type.value
            output.put({
                "is_pure_peer": result.local.num_ready_matched_blocks == 0,
                "physical": peer.physical_blocks.tolist(),
                "peer_node_id": peer.source.node_id,
                "staged_type": staged_type,
                "direct_type": direct_type,
            })
            engine.unlock(peer.pre_locked_node)
            done.set()
    except Exception:
        output.put({"error": traceback.format_exc(), "rank": rank})
        ready.set()
        done.set()


@pytest.mark.skipif(
    os.getenv("FLEXKV_RUN_RADIX_PEER_TEST") != "1",
    reason="set FLEXKV_RUN_RADIX_PEER_TEST=1 to run the RDMA test",
)
def test_radixshmem_single_peer_match_over_rdma():
    ctx = mp.get_context("spawn")
    ready = ctx.Event()
    done = ctx.Event()
    output = ctx.Queue()
    prefix = f"/flexkv_radix_peer_test_{os.getpid()}"
    port = int(os.getenv("FLEXKV_TEST_RADIX_PORT", "19600"))
    rdma_dev = os.getenv("FLEXKV_TEST_RDMA_DEVICES", "mlx5_0").split(",")[0]

    processes = [
        ctx.Process(
            target=_rank_main,
            args=(rank, prefix, port, rdma_dev, ready, done, output),
        )
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=40)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)

    messages = []
    while not output.empty():
        messages.append(output.get())
    errors = [message for message in messages if "error" in message]
    assert not errors, errors
    assert all(process.exitcode == 0 for process in processes)
    result = next(message for message in messages if "is_pure_peer" in message)
    assert result["is_pure_peer"] is True
    assert result["peer_node_id"] == 100
    assert result["staged_type"] == "PEERH2H"
    assert result["direct_type"] == "PEERH2D"
