
import json
import os
import torch
import tempfile
from typing import TYPE_CHECKING
from dataclasses import dataclass, field

from flexkv.common.debug import flexkv_logger
from flexkv.common.config import *

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import KVCacheConfig, FullAttentionSpec
    from vllm.config import VllmConfig


logger = flexkv_logger


def _parse_dtype_str(dtype_str: str) -> torch.dtype:
    """Convert a dtype string (e.g. 'fp8', 'bfloat16', 'fp8_e4m3') to torch.dtype.

    Shared by sglang / vllm / TRT-LLM integration adapters so that dtype
    parsing logic is defined in exactly one place.
    """
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
        "fp8": torch.float8_e4m3fn,
        "float8": torch.float8_e4m3fn,
        "e4m3": torch.float8_e4m3fn,
        "fp8_e4m3": torch.float8_e4m3fn,
    }
    return dtype_map.get(dtype_str.lower(), torch.bfloat16)


@dataclass
class FlexKVConfig:
    enable_flexkv: bool = True

    #base config
    server_recv_port: str = ""

    gpu_register_port: str = ""

    # cache config
    cache_config: CacheConfig = field(default_factory=CacheConfig)

    # model config
    model_config: ModelConfig = field(default_factory=ModelConfig)

    # user config
    user_config: UserConfig = field(default_factory=UserConfig)

    def __post_init__(self):
        if self.server_recv_port == "":
            self.server_recv_port = GLOBAL_CONFIG_FROM_ENV.server_recv_port
        if self.gpu_register_port == "":
            self.gpu_register_port = self.server_recv_port + "_gpu_register"

    @classmethod
    def from_env(cls) -> 'FlexKVConfig':
        enable_flexkv = bool(int(os.getenv('ENABLE_FLEXKV', 1)))
        config_file_path = os.getenv('FLEXKV_CONFIG_PATH', None)
        if config_file_path is None:
            logger.info("No flexkv config file provided, please set FLEXKV_CONFIG_PATH environment variable.")
            logger.info("Loading flexkv config from environment variables.")
            user_config = load_user_config_from_env()
            return cls(enable_flexkv=enable_flexkv,
                       user_config=user_config)
        else:
            logger.info(f"Loading flexkv config from file: {config_file_path}")
            user_config = load_user_config_from_file(config_file_path)
            return cls(enable_flexkv=enable_flexkv,
                       user_config=user_config)

    def post_init_from_vllm_config(
        self,
        vllm_config: "VllmConfig",
        ):
        self.cache_config.tokens_per_block = vllm_config.cache_config.block_size

        self.model_config.num_layers = vllm_config.model_config.get_num_layers(vllm_config.parallel_config)
        self.model_config.head_size = vllm_config.model_config.get_head_size()
        user_dtype_str = self.user_config.kv_cache_dtype
        vllm_kv_cache_dtype = getattr(vllm_config.cache_config, 'cache_dtype', 'auto')
        if user_dtype_str is not None:
            self.model_config.dtype = _parse_dtype_str(user_dtype_str)
            logger.info(
                f"[FlexKV vllm] Using kv_cache_dtype from user_config: "
                f"'{user_dtype_str}' -> {self.model_config.dtype}"
            )
        elif isinstance(vllm_kv_cache_dtype, str) and vllm_kv_cache_dtype != 'auto':
            self.model_config.dtype = _parse_dtype_str(vllm_kv_cache_dtype)
            logger.info(
                f"[FlexKV vllm] Using kv_cache_dtype from vllm cache_config: "
                f"'{vllm_kv_cache_dtype}' -> {self.model_config.dtype}"
            )
        else:
            self.model_config.dtype = vllm_config.model_config.dtype
            logger.info(
                f"[FlexKV vllm] No explicit kv_cache_dtype, falling back to "
                f"vllm model dtype: {self.model_config.dtype}"
            )
        self.model_config.use_mla = vllm_config.model_config.is_deepseek_mla
        if not self.model_config.use_mla:
            self._detect_packed_kv(vllm_config, vllm_kv_cache_dtype)
        self.model_config.tp_size = vllm_config.parallel_config.tensor_parallel_size
        self.model_config.dp_size = vllm_config.parallel_config.data_parallel_size
        if self.model_config.use_mla:
            self.model_config.num_kv_heads = 1
        else:
            self.model_config.num_kv_heads = vllm_config.model_config.get_total_num_kv_heads()
        update_default_config_from_user_config(self.model_config, self.cache_config, self.user_config)
        self.server_recv_port = GLOBAL_CONFIG_FROM_ENV.server_recv_port
        self.gpu_register_port = self.server_recv_port + "_gpu_register"

    def _detect_packed_kv(
        self,
        vllm_config: "VllmConfig",
        vllm_kv_cache_dtype: str,
        ) -> None:
        """Set packed_kv when vLLM keeps K and V in one content dim.

        Asks the attention backend for its cache shape rather than inspecting a
        tensor, because the layout has to be known before the KV cache exists.
        A 4D shape means K/V are packed into the last dim, which doubles the
        effective head_size and removes the kv dim (see
        :class:`~flexkv.common.storage.KVCacheLayout`).

        Args:
            vllm_config (VllmConfig): the vLLM config to read the backend from.
            vllm_kv_cache_dtype (str): vLLM's ``cache_config.cache_dtype``.
        """
        try:
            from vllm.distributed.kv_transfer.kv_connector.utils import (
                get_current_attn_backend,
            )

            cache_shape = get_current_attn_backend(vllm_config).get_kv_cache_shape(
                1,
                self.cache_config.tokens_per_block,
                vllm_config.model_config.get_num_kv_heads(vllm_config.parallel_config),
                self.model_config.head_size,
                vllm_kv_cache_dtype,
            )
        except (AttributeError, ImportError, TypeError, ValueError) as e:
            logger.debug(
                f"[FlexKV vllm] packed K/V layout detection unavailable ({e}); "
                "assuming the legacy split K/V layout"
            )
            return

        if len(cache_shape) == 4:
            self.model_config.packed_kv = True
            self.model_config.head_size = cache_shape[-1]
            logger.info(
                f"[FlexKV vllm] packed K/V layout detected: shape={cache_shape}, "
                f"head_size folded to {self.model_config.head_size}"
            )

    def post_init_from_sglang_config(
        self,
        sglang_config,
        tp_size: int,
        page_size: int,
    ):
        """
        Initialize FlexKVConfig fields from sglang config.
        Args:
            sglang_config: sglang.srt.configs.model_config.ModelConfig-like object
            tp_size: tensor parallel size used by sglang
            page_size: KV block size (tokens per block) used by sglang
        """
        # cache config
        self.cache_config.tokens_per_block = int(page_size)

        self.model_config.num_layers = int(getattr(sglang_config, "num_hidden_layers", 0))

        if hasattr(sglang_config, "get_num_kv_heads"):
            try:
                self.model_config.num_kv_heads = int(sglang_config.get_num_kv_heads(tp_size))
            except Exception:
                self.model_config.num_kv_heads = int(getattr(sglang_config, "num_key_value_heads", 0))
        else:
            self.model_config.num_kv_heads = int(getattr(sglang_config, "num_key_value_heads", 0))
        self.model_config.head_size = int(getattr(sglang_config, "head_dim", 0))

        # Determine KV cache dtype: prioritize user_config.kv_cache_dtype (from
        # flexkv_config.yaml or FLEXKV_KV_CACHE_DTYPE env var), then fall back to
        # the sglang model dtype.  sglang's ModelConfig.dtype is the *model
        # weight* dtype (e.g. bfloat16), which may differ from the KV cache dtype
        # (e.g. fp8_e4m3 when --kv-cache-dtype fp8_e4m3 is used).
        user_dtype_str = self.user_config.kv_cache_dtype
        if user_dtype_str is not None:
            self.model_config.dtype = _parse_dtype_str(user_dtype_str)
            logger.info(
                f"[FlexKV] Using kv_cache_dtype from user_config: "
                f"'{user_dtype_str}' -> {self.model_config.dtype}"
            )
        else:
            self.model_config.dtype = getattr(sglang_config, "dtype", torch.bfloat16)
            logger.warning(
                f"[FlexKV] No kv_cache_dtype in user_config, falling back to sglang "
                f"model dtype: {self.model_config.dtype}. If your KV cache uses a "
                f"different dtype (e.g. fp8), add 'kv_cache_dtype: fp8' to your "
                f"flexkv_config.yaml or set FLEXKV_KV_CACHE_DTYPE=fp8 environment variable."
            )

        attn_arch = getattr(sglang_config, "attention_arch", None)
        use_mla = False
        if hasattr(attn_arch, "name"):
            use_mla = (attn_arch.name.upper() == "MLA")
        elif isinstance(attn_arch, str):
            use_mla = (attn_arch.upper() == "MLA")
        self.model_config.use_mla = use_mla

        self.model_config.tp_size = int(tp_size)
        self.model_config.dp_size = int(getattr(sglang_config, "dp_size", 1))
        update_default_config_from_user_config(self.model_config, self.cache_config, self.user_config)

    def post_init_from_trt_config(
        self,
        config,
    ):
        self.cache_config.tokens_per_block = config.tokens_per_block
        # Convert dtype string to torch.dtype
        dtype_str = config.pytorch_backend_config.kv_cache_dtype
        flexkv_logger.info(f"[FlexKVConfig] dtype_str from TRT config: {dtype_str}")

        if dtype_str == "auto":
            # When dtype_str is "auto", try to get kv_cache_dtype from user_config first
            # This allows users to specify kv_cache_dtype in flexkv_config.json or via environment variable
            user_dtype_str = self.user_config.kv_cache_dtype
            if user_dtype_str is not None:
                parsed_dtype = _parse_dtype_str(user_dtype_str)
                self.model_config.dtype = parsed_dtype
                flexkv_logger.info(f"[FlexKVConfig] dtype_str='auto', but found kv_cache_dtype='{user_dtype_str}' in user_config, using it -> {parsed_dtype}")
            else:
                # Try to infer from TRT config if possible (e.g., from actual tensor dtype)
                # Note: This might not be available at initialization time
                self.model_config.dtype = torch.bfloat16
                flexkv_logger.warning(
                    f"[FlexKVConfig] dtype_str='auto' and no kv_cache_dtype in user_config. "
                    f"Falling back to {self.model_config.dtype}. To specify a different dtype, add 'kv_cache_dtype' "
                    f"to your flexkv_config.json file (e.g., {{\"kv_cache_dtype\": \"fp8\"}}) "
                    f"or set FLEXKV_KV_CACHE_DTYPE environment variable."
                )
        elif isinstance(dtype_str, str):
            self.model_config.dtype = _parse_dtype_str(dtype_str)
        else:
            self.model_config.dtype = dtype_str
        
        # Set model config (parallel configs part)
        if config.mapping.enable_attention_dp:
            self.model_config.tp_size = 1
            self.model_config.dp_size = config.mapping.tp_size
        else:
            self.model_config.tp_size = config.mapping.tp_size
            self.model_config.dp_size = 1
            
        # self.model_config (model configs part)
        try:
            model_path = getattr(config, 'hf_model_dir', None)
            from transformers import AutoConfig as HFAutoConfig
            hf_config = HFAutoConfig.from_pretrained(
                str(model_path), 
                trust_remote_code=True
            )
            self.model_config.num_layers = hf_config.num_hidden_layers
            self.model_config.use_mla = (hasattr(hf_config, 'kv_lora_rank') and 
                            hf_config.kv_lora_rank is not None and
                            hasattr(hf_config, 'qk_rope_head_dim') and 
                            hf_config.qk_rope_head_dim is not None)
            if self.model_config.use_mla:
                self.model_config.head_size = hf_config.kv_lora_rank + hf_config.qk_rope_head_dim
                self.model_config.num_kv_heads = 1
            else:
                if hasattr(hf_config, 'num_key_value_heads'):
                    assert hf_config.num_attention_heads != hf_config.num_key_value_heads, f"{hf_config.num_attention_heads=}, {hf_config.num_key_value_heads=}"
                    self.model_config.head_size = hf_config.head_dim
                    self.model_config.num_kv_heads = hf_config.num_key_value_heads
                else:
                    self.model_config.head_size = hf_config.hidden_size // hf_config.num_attention_heads
                    self.model_config.num_kv_heads = hf_config.num_attention_heads
            
        except Exception as e:
            flexkv_logger.error(f"Failed to load config from {model_path}: {e}")
        # Update cache config with user config after model config is initialized
        update_default_config_from_user_config(self.model_config, self.cache_config, self.user_config)
