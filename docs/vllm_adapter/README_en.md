# Using FlexKV in vLLM

## 🎉 Important Update (March 17, 2026)

**FlexKV has been officially merged into the vLLM mainline on March 17, 2026 (PR [#34328](https://github.com/vllm-project/vllm/pull/34328)).**

Starting from **vLLM v0.17.2rc0** (commit [`8cb24d3`](https://github.com/vllm-project/vllm/commit/8cb24d3aedb9f431fb15a636a3e11a00262f5991)), `FlexKVConnectorV1` is built into vLLM — **no patch is needed anymore**.

> **Recommended**: Install vLLM v0.17.2 ~ v0.28.x, or v0.33.0 / v0.34.1 with [its patch](#patching-vllm). Since v0.29.0 the built-in connector can no longer start FlexKV.

---

## Current Version vs. Legacy Version
In commit [`0290841dce65ae9b036a23d733cf94e47e814934`](https://github.com/taco-project/FlexKV/commit/0290841dce65ae9b036a23d733cf94e47e814934), we introduced a major update:  
**FlexKV has transitioned from a client-server architecture to a library function that inference acceleration engines (such as vLLM) can directly invoke**, reducing inter-process communication overhead.

This change involves significant API adjustments. Therefore, please note:

- **Version >= `1.0.0`**: Use the **current version API**.
- **Version == `0.1.0`**: Supports the **legacy version API** only.

---

## Current Version (>= 1.0.0)

### Supported Versions
- FlexKV >= `1.0.0`
- vLLM `0.17.2` ~ `0.28.x`: FlexKVConnectorV1 is built in — no patch required
- vLLM `0.33.0`: the built-in FlexKVConnectorV1 must be patched, see [Patching vLLM](#patching-vllm)
- vLLM `0.34.1` (main [`d51a514`](https://github.com/vllm-project/vllm/commit/d51a5147112a31dfbed08cfa4b774c27c34ce90d), nightly `0.34.1rc1.dev96`; not 0.34.0): the built-in FlexKVConnectorV1 must be patched, see [Patching vLLM](#patching-vllm)

### Configuration

#### Example 1: CPU Offloading Only
Use 32GB of CPU memory as secondary cache.
```bash
unset FLEXKV_CONFIG_PATH
export FLEXKV_CPU_CACHE_GB=32
```
#### Example 2: SSD Offloading
Use 32GB of CPU memory and 1TB of SSD storage as secondary and tertiary cache respectively. (Assume the machine has two SSDs mounted at /data0 and /data1 respectively.)
```bash
# generate config
cat <<EOF > ./flexkv_config.yml
cpu_cache_gb: 32
ssd_cache_gb: 1024
ssd_cache_dir: /data0/flexkv_ssd/;/data1/flexkv_ssd/
enable_gds: false
EOF
export FLEXKV_CONFIG_PATH="./flexkv_config.yml"
```
#### Example 3: radixshmem as the CPU Tier
The CPU tier and its prefix index live in a [radixshmem](../radixshmem/config_en.md) `radix-server`, one per node; its `--data-bytes`, not `cpu_cache_gb`, sets the CPU cache size.
```bash
radix-server --name /flexkv --data-bytes 64G
export FLEXKV_ENABLE_RADIXSHMEM=1
```

> Note: The `flexkv_config.yml` configuration is provided as a simple example only. For full parameter options, please refer to [`docs/flexkv_config_reference/README_en.md`](../../docs/flexkv_config_reference/README_en.md)

### Running

1. Install vLLM, then patch it if its version needs it (see [Patching vLLM](#patching-vllm))
```bash
pip install 'vllm==0.33.0'
# or 0.34.1, the nightly wheel its patch applies to
pip install 'vllm==0.34.1rc1.dev96+g249e65290' \
    --extra-index-url https://wheels.vllm.ai/249e652909228b61bba2c409ca5ae6c84d236783
```

2. Install FlexKV
```bash
pip install flexkv  # or build from source: ./build.sh
```

3. offline test
```bash
# Set VLLM_DIR to a compatible vLLM checkout containing this example.
python "$VLLM_DIR/examples/disaggregated/flexkv_connector/prefix_caching_flexkv.py" --model /path/to/model
```

4. online serving
```bash
VLLM_USE_V1=1 python -m vllm.entrypoints.cli.main serve Qwen3/Qwen3-32B \
     --tensor-parallel-size 8 \
     --trust-remote-code \
     --port 30001 \
     --max-num-seqs 128 \
     --max-num-batched-tokens 8192 \
     --max_model_len 8192 \
     --max-seq-len-to-capture 8192 \
     --gpu-memory-utilization 0.8 \
     --enable-chunked-prefill \
     --enable-prefix-caching \
     --kv-transfer-config \
        '{"kv_connector":"FlexKVConnectorV1","kv_role":"kv_both"}'
```

### Experimental Layer-wise KV Loading

Layer-wise loading overlaps FlexKV H2D transfer for layer `N+1` with vLLM
compute for layer `N`, reducing TTFT when external KV transfer is on the
critical path.

```bash
VLLM_USE_V1=1 python -m vllm.entrypoints.cli.main serve MODEL \
  --kv-transfer-config \
  '{"kv_connector":"FlexKVConnectorV1","kv_role":"kv_both",
"kv_connector_extra_config":{"use_layerwise":true}}'
```

`use_layerwise=true` is required so vLLM selects PIECEWISE CUDA graph mode;
the per-layer eventfd waits cannot execute correctly inside a captured full
CUDA graph. For safety, setting `FLEXKV_ENABLE_LAYERWISE_TRANSFER=1` alone is
ignored by the vLLM adapter. It needs the vLLM 0.33.0 or 0.34.1 patch.

The feature currently pipelines GET/H2D only. PUT remains request-level async.
Context parallelism (`context_parallel_size > 1`) is not supported yet because
vLLM's runtime DCP/PCP rank has not been mapped into FlexKV's eventfd index.
GDS is also unsupported in this mode because the fused layer-wise graph uses
CPU/SSD/remote staging rather than direct `DISK2D` operations.
If a layer does not become ready within
`FLEXKV_LAYERWISE_WAIT_TIMEOUT_S` (default `60` seconds), the forward fails
instead of hanging indefinitely. The model's attention must wait for each layer
it reads; FlexKV refuses to start otherwise. DeepSeek-V4.1's does with
`examples/vllm_adaption/vllm_0_33_0-deepseek-v41-layerwise.patch` on vLLM 0.33.0,
or `examples/vllm_adaption/vllm_0_34_1-deepseek-v41-layerwise.patch` on vLLM 0.34.1.

### Patching vLLM

Apply the patch matching your vLLM version:

| vLLM Version | Patch File |
|---|---|
| 0.33.0 | `examples/vllm_adaption/vllm_0_33_0-flexkv-connector.patch` |
| 0.34.1 | `examples/vllm_adaption/vllm_0_34_1-flexkv-connector.patch` |

```bash
cd vllm
# Set FLEXKV_DIR and choose the patch matching your vLLM version.
PATCH_FILE=vllm_0_34_1-flexkv-connector.patch
git apply "$FLEXKV_DIR/examples/vllm_adaption/$PATCH_FILE"
pip install -e .
```

The 0.33.0 and 0.34.1 patches make the built-in connector pass vLLM's
`KVCacheConfig` to FlexKV and forward cache reset, sleep/wake and layer-wise loading. It also lets
the connector run under vLLM's hybrid KV cache manager, which DeepSeek-V4.1
needs. With `--no-swa-bounded-replay` FlexKV also stores the sliding-window KV
at the last block boundary of each prompt, and a hit ends at such a boundary;
`swa.num_cpu_slots` (default 1024) and `swa.num_ssd_slots` in the FlexKV config
set how many the CPU and SSD tiers keep. The layers FlexKV stores may keep
pages of different sizes: a CPU/SSD block then holds one copy of the
pages when every TP rank holds the same ones (MLA, or vLLM's `kv_tp_replicas`
says so), and one copy per TP rank otherwise (e.g. GQA/MHA, whose heads TP
splits). On 0.34.1 the patch also keeps vLLM's extensible KV cache off, as
FlexKV's transfer engine opens the KV cache through CUDA IPC. A patch also
applies to an installed wheel:
```bash
cd "$(python -c 'import os, vllm; print(os.path.dirname(os.path.dirname(vllm.__file__)))')"
git apply --include='vllm/*' "$FLEXKV_DIR/examples/vllm_adaption/$PATCH_FILE"
```

With the 0.33.0 or 0.34.1 patch a FlexKV load that fails reaches vLLM as a failed receive,
and vLLM applies its `kv_load_failure_policy`: by default (`"fail"`) the request
ends with an error, with `"recompute"` vLLM recomputes it without FlexKV. Set it
in `--kv-transfer-config`, e.g.
`'{"kv_connector":"FlexKVConnectorV1","kv_role":"kv_both","kv_load_failure_policy":"recompute"}'`.
A layer-wise load that fails fails the forward instead (see above).

While FlexKV still copies a finished request's KV out of its GPU blocks, a vLLM
short of blocks waits for them rather than preempting a running request. It
stops waiting once the oldest such copy has run for `FLEXKV_PENDING_FREE_WAIT_S`
seconds (default `5`; `0` turns the wait off).
