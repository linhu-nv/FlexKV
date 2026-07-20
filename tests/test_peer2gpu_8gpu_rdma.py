"""Opt-in Mooncake RDMA smoke test using eight local GPUs as two nodes.

Run with a reachable Redis metadata service and RDMA device list, for example:

FLEXKV_RUN_8GPU_PEER_TEST=1 \
FLEXKV_TEST_ENGINE_IP=10.0.0.10 \
FLEXKV_TEST_RDMA_DEVICES=mlx5_0,mlx5_1 \
FLEXKV_TEST_MOONCAKE_METADATA=redis://127.0.0.1:6380 \
WITH_NVIDIA_PEERMEM=0 \
pytest -q tests/test_peer2gpu_8gpu_rdma.py -s

Set ``WITH_NVIDIA_PEERMEM=0`` when the host uses CUDA dma-buf registration
instead of the optional nvidia-peermem kernel module.
"""

import multiprocessing as mp
import os

import pytest


def _source_node(config, ready_queue, stop_event, elements_per_gpu):
    import torch

    from flexkv.common.config import MooncakeTransferEngineConfig
    from flexkv.mooncakeEngineWrapper import MoonCakeTransferEngineWrapper

    engine = MoonCakeTransferEngineWrapper(
        MooncakeTransferEngineConfig(**config)
    )
    source = torch.arange(
        4 * elements_per_gpu, dtype=torch.float32, device="cpu"
    )
    assert engine.regist_buffer(
        source.data_ptr(), source.numel() * source.element_size()
    ) == 0
    # Logical node 0 owns GPUs 0-3. Register them as this is the same GPU MR
    # setup performed by PEER2GPUTransferWorker on every participating node.
    owned_gpus = []
    for device_id in range(4):
        marker = torch.zeros(4096, dtype=torch.uint8, device=f"cuda:{device_id}")
        assert engine.regist_buffer(marker.data_ptr(), marker.numel()) == 0, (
            f"Mooncake could not register cuda:{device_id}; verify that "
            "nvidia_peermem is loaded, or set WITH_NVIDIA_PEERMEM=0 to use "
            "ibv_reg_dmabuf_mr"
        )
        owned_gpus.append(marker)
    ready_queue.put((engine.get_engine_addr(), source.data_ptr()))
    stop_event.wait(timeout=120)
    for marker in owned_gpus:
        engine.unregist_buffer(marker.data_ptr())
    engine.unregist_buffer(source.data_ptr())


def test_two_logical_nodes_peer_cpu_to_four_gpus_each_over_rdma():
    if os.getenv("FLEXKV_RUN_8GPU_PEER_TEST") != "1":
        pytest.skip("set FLEXKV_RUN_8GPU_PEER_TEST=1 to run the 8-GPU RDMA test")

    import torch

    if torch.cuda.device_count() < 8:
        pytest.skip("the local two-node simulation requires at least 8 GPUs")

    engine_ip = os.environ["FLEXKV_TEST_ENGINE_IP"]
    rdma_devices = os.environ["FLEXKV_TEST_RDMA_DEVICES"]
    rdma_device_path = os.getenv(
        "FLEXKV_TEST_RDMA_DEVICE_PATH", "/dev/infiniband/uverbs0"
    )
    try:
        rdma_fd = os.open(rdma_device_path, os.O_RDWR)
    except OSError as exc:
        pytest.fail(
            f"RDMA device {rdma_device_path} is not accessible: {exc}; "
            "pass /dev/infiniband devices through the container runtime"
        )
    else:
        os.close(rdma_fd)
    metadata_server = os.environ["FLEXKV_TEST_MOONCAKE_METADATA"]
    metadata_backend = os.getenv("FLEXKV_TEST_MOONCAKE_BACKEND", "redis")
    metadata_auth = os.getenv("FLEXKV_TEST_MOONCAKE_AUTH", "")
    base_port = int(os.getenv("FLEXKV_TEST_MOONCAKE_PORT", "19500"))
    elements_per_gpu = 4096

    common = dict(
        engine_ip=engine_ip,
        metadata_backend=metadata_backend,
        metadata_server=metadata_server,
        metadata_server_auth=metadata_auth,
        protocol="rdma",
        device_name=rdma_devices,
    )
    source_config = dict(common, engine_port=base_port)
    destination_config = dict(common, engine_port=base_port + 1)

    ctx = mp.get_context("spawn")
    ready_queue = ctx.Queue()
    stop_event = ctx.Event()
    source_process = ctx.Process(
        target=_source_node,
        args=(source_config, ready_queue, stop_event, elements_per_gpu),
    )
    source_process.start()

    from flexkv.common.config import MooncakeTransferEngineConfig
    from flexkv.mooncakeEngineWrapper import MoonCakeTransferEngineWrapper

    destination_engine = None
    destinations = []
    try:
        source_addr, source_ptr = ready_queue.get(timeout=60)
        destination_engine = MoonCakeTransferEngineWrapper(
            MooncakeTransferEngineConfig(**destination_config)
        )
        for device_id in range(4, 8):
            with torch.cuda.device(device_id):
                tensor = torch.empty(
                    elements_per_gpu, dtype=torch.float32, device=f"cuda:{device_id}"
                )
            assert destination_engine.regist_buffer(
                tensor.data_ptr(), tensor.numel() * tensor.element_size()
            ) == 0, (
                f"Mooncake could not register cuda:{device_id}; verify that "
                "nvidia_peermem is loaded, or set WITH_NVIDIA_PEERMEM=0 to "
                "use ibv_reg_dmabuf_mr"
            )
            destinations.append(tensor)

        bytes_per_gpu = elements_per_gpu * torch.float32.itemsize
        ret = destination_engine.batch_transfer_sync_read(
            source_addr,
            [source_ptr + rank * bytes_per_gpu for rank in range(4)],
            [tensor.data_ptr() for tensor in destinations],
            [bytes_per_gpu] * 4,
        )
        assert ret == 0
        for rank, tensor in enumerate(destinations):
            expected = torch.arange(
                rank * elements_per_gpu,
                (rank + 1) * elements_per_gpu,
                dtype=torch.float32,
                device=tensor.device,
            )
            torch.testing.assert_close(tensor, expected)
    finally:
        if destination_engine is not None:
            for tensor in destinations:
                destination_engine.unregist_buffer(tensor.data_ptr())
        stop_event.set()
        source_process.join(timeout=15)
        if source_process.is_alive():
            source_process.terminate()
            source_process.join(timeout=5)
