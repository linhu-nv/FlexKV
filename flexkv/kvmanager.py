# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import subprocess
from typing import Optional, Tuple, List, Dict, Union, Iterable
import time

import numpy as np
import torch

from flexkv.server.client import KVDPClient
from flexkv.server.server import KVServer, DPClient
from flexkv.kvtask import KVTaskEngine, KVResponse
from flexkv.common.config import ModelConfig, CacheConfig, GLOBAL_CONFIG_FROM_ENV, MooncakeTransferEngineConfig
from flexkv.common.transfer import TransferOpGraph
from flexkv.integration.dynamo.collector import KVEventCollector
from flexkv.common.debug import flexkv_logger
from flexkv.cache.redis_meta import RedisMeta


class KVManager:
    def __init__(self,
                 model_config: ModelConfig,
                 cache_config: CacheConfig,
                 dp_client_id: int = 0,
                 server_recv_port: str = "",
                 gpu_register_port: str = "",
                 event_collector: Optional[KVEventCollector] = None):
        flexkv_logger.info(f"{model_config = }")
        flexkv_logger.info(f"{cache_config = }")
        flexkv_logger.info(f"{GLOBAL_CONFIG_FROM_ENV = }")
        self.model_config = model_config
        self.cache_config = cache_config
        self.instance_id = GLOBAL_CONFIG_FROM_ENV.instance_id
        self.instance_num = GLOBAL_CONFIG_FROM_ENV.instance_num

        if server_recv_port != "":
            self.server_recv_port = server_recv_port
        else:
            self.server_recv_port = GLOBAL_CONFIG_FROM_ENV.server_recv_port
        if gpu_register_port != "":
            self.gpu_register_port = gpu_register_port
        else:
            self.gpu_register_port = self.server_recv_port + "_gpu_register"

        # When radix_shmem is enabled we don't need a separate KVServer at
        # all — the index lives in shared memory and each DP scheduler proc
        # owns its own KVTaskEngine that talks to a single shared TE process
        # via shm channels.
        self.use_radix_shmem = bool(getattr(GLOBAL_CONFIG_FROM_ENV,
                                            "radix_shmem", False))
        self._shm_radix_server_id = getattr(GLOBAL_CONFIG_FROM_ENV,
                                            "shm_radix_server_id", "default")

        # Multi-instance mode also requires server_client_mode
        if self.use_radix_shmem:
            # Force server_client_mode False — KVServer is bypassed entirely.
            self.server_client_mode = False
        else:
            self.server_client_mode = (model_config.dp_size > 1 or
                                       self.instance_num > 1 or
                                       GLOBAL_CONFIG_FROM_ENV.server_client_mode)
        self.dp_client_id = dp_client_id

        # Calculate global_client_id for multi-instance mode
        self.global_client_id = self.instance_id * model_config.dp_size + dp_client_id

        flexkv_logger.info(
            f"server_client_mode: {self.server_client_mode}, "
            f"use_radix_shmem: {self.use_radix_shmem}"
        )

        self.redis_meta_client = None
        self.enable_mps = GLOBAL_CONFIG_FROM_ENV.enable_mps
        # Owner handle for shm radix regions — only the bootstrap process
        # holds this; others have None.
        self._shm_radix_owners = None
        # TE-process handle — only the bootstrap process holds this.
        self._shm_te_process = None
        # Local KVTaskEngine for the radix-shmem path (per-DP).
        self.kv_task_engine = None
        self.server_handle = None

        if self.use_radix_shmem:
            self._init_radix_shmem_path(event_collector)
        elif self.server_client_mode:
            # In server_client_mode, RedisMeta is created and initialized inside KVServer
            # Server should only be created once across all instances and dp ranks
            if self.instance_id == 0 and dp_client_id == 0:
                total_clients = self.instance_num * model_config.dp_size
                self.server_handle = KVServer.create_server(model_config=model_config,
                                                            cache_config=cache_config,
                                                            gpu_register_port=self.gpu_register_port,
                                                            server_recv_port=self.server_recv_port,
                                                            total_clients=total_clients,
                                                            inherit_env=False)

            else:
                self.server_handle = None
            self.dp_client = KVDPClient(self.server_recv_port, self.model_config, self.global_client_id)
        else:
            # In non-server_client_mode, create RedisMeta here and pass to KVTaskEngine
            if self.cache_config.enable_kv_sharing:
                flexkv_logger.info(f"[kv manager] initializing RedisMeta and connection to "
                                   f"{self.cache_config.redis_host}:{self.cache_config.redis_port}")
                self.redis_meta_client = RedisMeta(
                    self.cache_config.redis_host,
                    self.cache_config.redis_port,
                    self.cache_config.redis_password,
                    self.cache_config.local_ip,
                    node_ttl_seconds=self.cache_config.node_ttl_seconds,
                )
                self.redis_meta_client.init_meta()
                # update distributed_node_id
                self.cache_config.distributed_node_id = self.redis_meta_client.get_node_id()

            self.server_handle = None
            self.kv_task_engine = KVTaskEngine(self.model_config, self.cache_config, self.gpu_register_port, redis_meta=self.redis_meta_client, event_collector=event_collector)

    def _init_radix_shmem_path(self,
                               event_collector: Optional[KVEventCollector]) -> None:
        """Initialize the radix-shmem multi-DP path.

        Bootstrap proc (instance 0, dp 0):
          1. Create radix shm regions (one TreeServer per device type).
          2. Spawn the single TE subprocess (which creates per-CE shm channels).

        Other DP procs:
          1. Poll for the radix shm regions (created by bootstrap).
          2. Continue — TE channel attach blocks on `ShmControlBlock.wait_ready`
             inside `TransferManagerShmChannelHandle`.

        Each CE process gets a disjoint graph_id range so submissions to the
        single shared TE never collide.
        """
        from flexkv.server.shm_radix_bootstrap import (
            create_shm_radix_regions, attach_shm_radix_clients,
        )
        from flexkv.transfer_manager import TransferManagerShmTEProcess

        is_bootstrap = (self.instance_id == 0 and self.dp_client_id == 0)
        total_clients = self.instance_num * self.model_config.dp_size
        server_id = self._shm_radix_server_id

        # Disjoint graph_id and op_id ranges per CE process: 2^32 ids per CE,
        # high bits = global_client_id. Critical for the multi-DP path where
        # all CE procs feed a single TE that uses op_id as a primary key for
        # internal bookkeeping (op_id_to_op, op_id_to_nvtx_range, etc.).
        from flexkv.common.transfer import TransferOp
        TransferOpGraph.set_graph_id_range(
            self.global_client_id << 32,
            (self.global_client_id + 1) << 32,
        )
        TransferOp.set_op_id_range(
            self.global_client_id << 32,
            (self.global_client_id + 1) << 32,
        )

        if is_bootstrap:
            self._shm_radix_owners = create_shm_radix_regions(
                self.model_config, self.cache_config,
                server_id=server_id,
            )
            self._shm_te_process = TransferManagerShmTEProcess(
                self.model_config, self.cache_config,
                gpu_register_port=self.gpu_register_port,
                server_id=server_id,
                num_channels=total_clients,
            )
            self._shm_te_process.start()
        else:
            # Wait until bootstrap created the shm radix regions before
            # GlobalCacheEngine tries to attach as TreeClient.
            attach_shm_radix_clients(self.cache_config, server_id=server_id)

        # GlobalCacheEngine inspects GLOBAL_CONFIG_FROM_ENV.radix_shmem and
        # constructs CacheEngineRadixShmem (TreeClient) per device type.
        # KVTaskEngine wires up the shm-mode TransferManagerHandle.
        self.kv_task_engine = KVTaskEngine(
            self.model_config, self.cache_config,
            self.gpu_register_port,
            redis_meta=self.redis_meta_client,
            event_collector=event_collector,
            shm_te_server_id=server_id,
            shm_te_channel_id=self.global_client_id,
        )

    @property
    def dpclient_id(self) -> int:
        return self.dp_client_id

    def start(self) -> None:
        if self.enable_mps:
            # try to start MPS
            subprocess.run(['nvidia-cuda-mps-control', '-d'], check=False)
            flexkv_logger.debug("MPS started")

        if not self.server_client_mode:
            self.kv_task_engine.start()
        else:
            # send the start request to the server
            self.dp_client.start_server_and_register()

    def is_ready(self) -> bool:
        if self.server_client_mode:
            return self.dp_client.is_ready()
        else:
            return self.kv_task_engine.is_ready()

    def shutdown(self) -> None:
        if self.server_client_mode:
            self.dp_client.shutdown()
            # Wait for the server process to exit after sending shutdown request
            if self.server_handle is not None:
                self.server_handle.shutdown()
                self.server_handle = None
        else:
            if self.kv_task_engine is not None:
                self.kv_task_engine.shutdown()

        # Multi-DP radix-shmem teardown — only the bootstrap proc owns these.
        if self._shm_te_process is not None:
            self._shm_te_process.shutdown()
            self._shm_te_process = None
        if self._shm_radix_owners is not None:
            self._shm_radix_owners.shutdown()
            self._shm_radix_owners = None

        if self.enable_mps:
            flexkv_logger.info(
                "MPS is enabled. To stop MPS daemon manually, run: "
                "'echo quit | nvidia-cuda-mps-control'"
            )

    def get_async(self,
                  token_ids: Union[torch.Tensor, np.ndarray],
                  slot_mapping: Union[torch.Tensor, np.ndarray],
                  token_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
                  layer_granularity: int = -1,
                  dp_id: int = 0,
                  namespace: Optional[List[str]] = None,
                  ) -> int:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.numpy()
        if isinstance(slot_mapping, torch.Tensor):
            slot_mapping = slot_mapping.numpy()
        if isinstance(token_mask, torch.Tensor):
            token_mask = token_mask.numpy()
        if self.server_client_mode:
            task_id = self.dp_client.get_async(token_ids,
                                               slot_mapping,
                                               token_mask,
                                               layer_granularity,
                                               namespace=namespace)
        else:
            task_id, _ = self.kv_task_engine.get_async(token_ids,
                                                       slot_mapping,
                                                       token_mask,
                                                       layer_granularity,
                                                       dp_id,
                                                       namespace=namespace)
        return task_id

    def get_match(self,
                  token_ids: Union[torch.Tensor, np.ndarray],
                  token_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
                  layer_granularity: int = -1,
                  dp_id: int = 0,
                  cpu_only: bool = False,
                  namespace: Optional[List[str]] = None,
                  ) -> Tuple[int, np.ndarray]:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.numpy()
        if isinstance(token_mask, torch.Tensor):
            token_mask = token_mask.numpy()
        if self.server_client_mode:
            task_id, mask = self.dp_client.get_match(token_ids,
                                                     token_mask,
                                                     layer_granularity,
                                                     cpu_only=cpu_only,
                                                     namespace=namespace)
        else:
            task_id, mask = self.kv_task_engine.get_match(token_ids,
                                                          token_mask,
                                                          layer_granularity,
                                                          dp_id,
                                                          cpu_only=cpu_only,
                                                          namespace=namespace)
        return task_id, mask

    def put_async(self,
                  token_ids: Union[torch.Tensor, np.ndarray],
                  slot_mapping: Union[torch.Tensor, np.ndarray],
                  token_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
                  dp_id: int = 0,
                  namespace: Optional[List[str]] = None,
                  ) -> int:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.numpy()
        if isinstance(slot_mapping, torch.Tensor):
            slot_mapping = slot_mapping.numpy()
        if isinstance(token_mask, torch.Tensor):
            token_mask = token_mask.numpy()
        if self.server_client_mode:
            task_id = self.dp_client.put_async(token_ids, slot_mapping, token_mask, namespace=namespace)
        else:
            task_id, _ = self.kv_task_engine.put_async(token_ids, slot_mapping, token_mask, dp_id, namespace=namespace)
        return task_id

    def put_match(self,
                  token_ids: Union[torch.Tensor, np.ndarray],
                  token_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
                  dp_id: int = 0,
                  namespace: Optional[List[str]] = None,
                  ) -> Tuple[int, np.ndarray]:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.numpy()
        if isinstance(token_mask, torch.Tensor):
            token_mask = token_mask.numpy()
        if self.server_client_mode:
            task_id, mask = self.dp_client.put_match(token_ids, token_mask, namespace=namespace)
        else:
            task_id, mask = self.kv_task_engine.put_match(token_ids, token_mask, dp_id, namespace=namespace)
        return task_id, mask

    def prefetch_async(self,
                       token_ids: np.ndarray,
                       dp_id: int = 0,
                       namespace: Optional[List[str]] = None) -> int:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.numpy()
        if self.server_client_mode:
            task_id = self.dp_client.prefetch_async(token_ids, namespace=namespace)
        else:
            task_id = self.kv_task_engine.prefetch_async(token_ids, dp_id=dp_id, namespace=namespace)
        return task_id

    def launch(self,
               task_ids: Union[int, List[int]],
               slot_mappings: Union[np.ndarray, List[np.ndarray], torch.Tensor, List[torch.Tensor]],
               as_batch: bool = False) -> List[int]:
        if isinstance(task_ids, int):
            task_ids = [task_ids]
        if not isinstance(slot_mappings, List):
            slot_mappings = [slot_mappings]
        if isinstance(slot_mappings[0], torch.Tensor):
            slot_mappings = [slot_mapping.numpy() for slot_mapping in slot_mappings]
        if self.server_client_mode:
            return self.dp_client.launch_tasks(task_ids, slot_mappings, as_batch)
        else:
            return self.kv_task_engine.launch_tasks(task_ids, slot_mappings, as_batch)

    def cancel(self, task_ids: Union[int, List[int]]) -> None:
        if isinstance(task_ids, int):
            task_ids = [task_ids]
        if self.server_client_mode:
            self.dp_client.cancel_task(task_ids)
        else:
            self.kv_task_engine.cancel_tasks(task_ids)

    def wait(self,
             task_ids: Union[int, List[int]],
             timeout: float = 20.0,
             completely: bool = False) -> Dict[int, KVResponse]:
        if isinstance(task_ids, int):
            task_ids = [task_ids]
        if self.server_client_mode:
            return self.dp_client.wait(task_ids, timeout, completely)
        else:
            return self.kv_task_engine.wait(task_ids, timeout, completely)

    def try_wait(self, task_ids: Union[int, List[int]]) -> Dict[int, KVResponse]:
        if isinstance(task_ids, int):
            task_ids = [task_ids]
        if self.server_client_mode:
            return self.dp_client.try_wait(task_ids)
        else:
            return self.kv_task_engine.try_wait(task_ids)

    # Only for testing
    def _clear_cpu_cache(self) -> None:
        if self.server_client_mode:
            flexkv_logger.error("clear_cache is not supported in server client mode")
            return
        else:
            self.kv_task_engine._clear_cpu_cache()
