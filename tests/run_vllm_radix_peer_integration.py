"""Opt-in dual-vLLM peer-reuse integration runner.

Drives two vLLM instances that share a prefix through FlexKV's distributed
peer path.  ``FLEXKV_TEST_RADIX_SHMEM=1`` (default) exercises the RDMA
``CacheEngineRadixShmem``; ``=0`` exercises the Redis+Mooncake
``HierarchyLRCacheEngine``.  Both funnel through the same refactored GET code
(``get_media_list`` / ``_get_impl_without_lake`` / flat locking).

Run under /root/env.sh with two available GPUs.  The source request is
followed by an extra source-side request to drive connector polling before the
same prefix is sent to the target vLLM.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path


_FATAL_STARTUP_MARKERS = (
    "Mooncake failed to register GPU buffer",
    "Address already in use",
    "cudaHostRegister failed",
)


def _request(port: int, model: str, prompt: str) -> dict:
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "max_tokens": 8,
        "temperature": 0,
    }).encode()
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.load(response)


def _wait_ready(
    process: subprocess.Popen,
    port: int,
    timeout: int,
    log_path: Path,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"vLLM on port {port} exited with {process.returncode}")
        startup_log = log_path.read_text(errors="replace")
        if marker := next(
            (item for item in _FATAL_STARTUP_MARKERS if item in startup_log),
            None,
        ):
            raise RuntimeError(
                f"vLLM on port {port} failed during startup: {marker}"
            )
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/health", timeout=2
            ):
                return
        except Exception:
            time.sleep(2)
    raise TimeoutError(f"vLLM on port {port} was not ready in {timeout}s")


def main() -> int:
    model = os.getenv("FLEXKV_TEST_MODEL", "Qwen/Qwen3-0.6B")
    direct_gpu = os.getenv("FLEXKV_TEST_DIRECT_GPU", "0") == "1"
    peer_tier = os.getenv("FLEXKV_TEST_PEER_TIER", "cpu").lower()
    if peer_tier not in {"cpu", "ssd"}:
        raise ValueError("FLEXKV_TEST_PEER_TIER must be 'cpu' or 'ssd'")
    if peer_tier == "ssd":
        expected_route = "PEERSSD2D" if direct_gpu else "PEERSSD2H"
    else:
        expected_route = "PEERH2D" if direct_gpu else "PEERH2H"
    # RADIX_SHMEM=1 exercises CacheEngineRadixShmem (RDMA distributed radix);
    # =0 exercises HierarchyLRCacheEngine (Redis meta + Mooncake data path).
    # Both share the refactored get_media_list/_get_impl_without_lake GET code.
    use_shmem = os.getenv("FLEXKV_TEST_RADIX_SHMEM", "1") == "1"
    base_port = int(os.getenv("FLEXKV_TEST_VLLM_PORT", "31120"))
    radix_port = int(os.getenv("FLEXKV_TEST_RADIX_PORT", "19720"))
    zmq_base_port = int(os.getenv(
        "FLEXKV_TEST_ZMQ_PORT", str(base_port + 20000)
    ))
    mooncake_base_port = int(os.getenv(
        "FLEXKV_TEST_MOONCAKE_PORT", str(base_port + 21000)
    ))
    cluster_id = f"vllm_peer_{os.getpid()}"
    ip = os.getenv("FLEXKV_TEST_ENGINE_IP", "10.57.204.227")
    rdma_dev = os.getenv("FLEXKV_TEST_RDMA_DEVICES", "mlx5_0").split(",")[0]
    metadata = os.getenv(
        "FLEXKV_MOONCAKE_META", "http://127.0.0.1:8080/metadata"
    )
    gpu_ids = [
        gpu.strip()
        for gpu in os.getenv("FLEXKV_TEST_GPUS", "2,3").split(",")
        if gpu.strip()
    ]
    if len(gpu_ids) != 2:
        raise ValueError("FLEXKV_TEST_GPUS must contain exactly two GPU IDs")
    processes = []
    logs = []

    with tempfile.TemporaryDirectory(prefix="flexkv-vllm-peer-") as temp:
        temp_path = Path(temp)
        mps_pipe_dir = temp_path / "mps-pipe"
        mps_log_dir = temp_path / "mps-log"
        mps_pipe_dir.mkdir()
        mps_log_dir.mkdir()
        try:
            for rank in range(2):
                use_peer_ssd = peer_tier == "ssd"
                user_config = {
                    "cpu_cache_gb": 1 if use_peer_ssd else 2,
                    "ssd_cache_gb": 2 if use_peer_ssd else 0,
                    "ssd_cache_dir": str(temp_path / f"ssd-rank{rank}"),
                    "enable_p2p_cpu": not use_peer_ssd,
                    "enable_p2p_ssd": use_peer_ssd,
                    "enable_p2p_gpu": direct_gpu,
                    "redis_host": os.getenv("FLEXKV_REDIS_HOST", "127.0.0.1"),
                    "redis_port": int(os.getenv("FLEXKV_REDIS_PORT", "6379")),
                    "local_ip": ip,
                    "local_zmq_ip": ip,
                    # PEERSSD2H uses local_zmq_port for metadata and the next
                    # port for completion status, so each rank needs a pair.
                    "local_zmq_port": zmq_base_port + rank * 2,
                    "node_ttl_seconds": 60,
                }
                config_path = temp_path / f"flexkv-rank{rank}.json"
                config_path.write_text(json.dumps(user_config))
                mooncake_path = temp_path / f"mooncake-rank{rank}.json"
                mooncake_path.write_text(json.dumps({
                    "engine_ip": ip,
                    "engine_port": mooncake_base_port + rank,
                    "metadata_backend": "http",
                    "metadata_server": metadata,
                    "metadata_server_auth": "",
                    "protocol": os.getenv("FLEXKV_MOONCAKE_PROTOCOL", "rdma"),
                    "device_name": rdma_dev,
                }))
                log_path = temp_path / f"vllm-rank{rank}.log"
                log_file = log_path.open("w")
                logs.append((log_path, log_file))

                env = os.environ.copy()
                env.update({
                    "CUDA_VISIBLE_DEVICES": gpu_ids[rank],
                    "CUDA_MPS_PIPE_DIRECTORY": str(mps_pipe_dir),
                    "CUDA_MPS_LOG_DIRECTORY": str(mps_log_dir),
                    "FLEXKV_RADIX_SHMEM": "1" if use_shmem else "0",
                    "FLEXKV_ENABLE_MPS": "1",
                    "FLEXKV_TRACE_RADIX_PEER": "1",
                    "FLEXKV_CONFIG_PATH": str(config_path),
                    "MOONCAKE_CONFIG_PATH": str(mooncake_path),
                    "FLEXKV_SERVER_RECV_PORT": (
                        f"ipc:///tmp/{cluster_id}_rank{rank}"
                    ),
                    "HF_HUB_OFFLINE": "1",
                })
                # Shmem needs the RDMA radix-tree bootstrap; hie discovers peers
                # and node ids through Redis, so those knobs are shmem-only.
                if use_shmem:
                    env.update({
                        "FLEXKV_SHM_RADIX_ID": cluster_id,
                        "FLEXKV_RADIX_CLUSTER_ID": cluster_id,
                        "FLEXKV_RADIX_RANK": str(rank),
                        "FLEXKV_RADIX_WORLD_SIZE": "2",
                        "FLEXKV_RADIX_MASTER_ADDR": "127.0.0.1",
                        "FLEXKV_RADIX_MASTER_PORT": str(radix_port),
                        "FLEXKV_RADIX_RDMA_DEV": rdma_dev,
                        "FLEXKV_RADIX_GID_IDX": "3",
                    })
                command = [
                    "vllm", "serve", model,
                    "--tensor-parallel-size", "1",
                    "--data-parallel-size", "1",
                    "--enforce-eager",
                    "--port", str(base_port + rank),
                    "--max-num-seqs", "8",
                    "--max-num-batched-tokens", "2048",
                    "--max-model-len", "2048",
                    "--gpu-memory-utilization", "0.25",
                    "--no-enable-prefix-caching",
                    "--trust-remote-code",
                    "--kv-transfer-config",
                    '{"kv_connector":"FlexKVConnectorV1","kv_role":"kv_both"}',
                ]
                processes.append(subprocess.Popen(
                    command,
                    env=env,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                ))

            for rank, process in enumerate(processes):
                _wait_ready(
                    process,
                    base_port + rank,
                    timeout=180,
                    log_path=logs[rank][0],
                )

            shared = (
                "FlexKV distributed prefix validation. "
                + "The quick brown fox jumps over the lazy dog. " * 40
                + "Summarize the preceding passage."
            )
            source = _request(base_port, model, shared)
            time.sleep(2)
            # Required connector polling turn on the source vLLM.
            _request(
                base_port,
                model,
                "Connector polling request after publishing the shared prefix.",
            )
            time.sleep(3)
            target = _request(base_port + 1, model, shared)
            time.sleep(5)

            for _, log_file in logs:
                log_file.flush()
            target_log = logs[1][0].read_text(errors="replace")
            peer_plan_lines = [
                line
                for line in target_log.splitlines()
                if "[PEER GET PLAN]" in line
            ]
            if not any(expected_route in line for line in peer_plan_lines):
                raise AssertionError(
                    f"target request plan did not contain {expected_route}; "
                    f"peer plans={peer_plan_lines!r}\n"
                    + target_log[-8000:]
                )
            source_text = source["choices"][0]["text"]
            target_text = target["choices"][0]["text"]
            if source_text != target_text:
                raise AssertionError(
                    f"source/target output mismatch: {source_text!r} != {target_text!r}"
                )
            print(
                f"PASS dual-vLLM radix peer route={expected_route} "
                f"output={target_text!r}"
            )
            return 0
        except Exception:
            for log_path, log_file in logs:
                log_file.flush()
                print(f"\n===== {log_path.name} =====")
                print(log_path.read_text(errors="replace")[-12000:])
            raise
        finally:
            for process in processes:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
            deadline = time.monotonic() + 20
            for process in processes:
                remaining = max(0, deadline - time.monotonic())
                try:
                    process.wait(timeout=remaining)
                except subprocess.TimeoutExpired:
                    pass
                # vLLM's API process can exit before its orphaned engine and
                # FlexKV worker descendants.  Always clear the whole test
                # process group after the graceful window.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                if process.poll() is None:
                    process.wait(timeout=5)
            for _, log_file in logs:
                log_file.close()
            mps_env = os.environ.copy()
            mps_env["CUDA_MPS_PIPE_DIRECTORY"] = str(mps_pipe_dir)
            mps_env["CUDA_MPS_LOG_DIRECTORY"] = str(mps_log_dir)
            subprocess.run(
                ["nvidia-cuda-mps-control"],
                input="quit\n",
                text=True,
                env=mps_env,
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )


if __name__ == "__main__":
    raise SystemExit(main())
