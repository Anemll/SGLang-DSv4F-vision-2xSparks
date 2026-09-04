# DeepSeek-V4-Flash-Vision-Exp on 2× DGX Spark (SGLang)

**Date:** 2026-09-04  
**Last updated:** 2026-09-04 12:45 PT  
**Cookbook:** https://docs.sglang.io/cookbook/autoregressive/DeepSeek/DeepSeek-V4.md  
**Weights (HF):** [`deepseek-ai/DeepSeek-V4-Flash-Vision-Exp`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp)

Notes from bringing Vision-Exp up on a 2× DGX Spark (GB10 UMA) pair with SGLang: official recipe → what broke → one-file fix → working launch.

Lab-only absolute paths (NAS / Mac / Spark hostnames) live in a **gitignored** local file (`LAB.md`) and are not published here.

---

## TL;DR

| Item | Cookbook / stock | What actually worked here |
| --- | --- | --- |
| Image | `lmsysorg/sglang:dev-v4f-2dgx-v2` | Same image **+ bind-mount** streamed `deepseek_v4_vl.py` |
| Weight load | Stock VL `load_weights` | **Must stream** — stock buffers all LLM tensors → ~2× peak RSS → OOM on GB10 |
| Shared-experts fusion | Engine auto-disables on Vision; recipe may still show fusion path | Explicit `--disable-shared-experts-fusion` |
| Page cache | Optional | `--weight-loader-drop-cache-after-load` (important on UMA) |
| Speculative | DSPARK on balanced Spark cell | DSPARK works **after** streaming patch (draft is `DeepseekV4ForCausalLMDSpark` ~5.7 GB, not a 2nd full VL) |
| Decode CUDA graphs | On | Works on `-v2` with DSPARK. Failed earlier on non-v2 thin overlay (`HashTopK` + `input_ids=None`) |
| Tool / reasoning parsers | Playground toggles | `--tool-call-parser deepseekv4` + `--reasoning-parser deepseek-v4` |
| Never use on GB10 | — | `--weight-loader-disable-mmap` (host hang / hard reboot) |

**Root cause of stock `-v2` OOM:**  
`deepseek_v4_vl.DeepseekV4ForCausalLM.load_weights` collected every non-vision tensor into a Python `list` before calling `language_model.load_weights`. On 128 GB UMA that peaks near **2×** weight footprint and the OS OOM-kills the scheduler (exit -9) after 48/48 shards, never reaching `Load weight end`.

**Fix:** stream via a generator (see `overlay/deepseek_v4_vl.py`). After that, load end ≈ **74–76 GB/rank**.

---

## Reproduce

### 1. Download weights from Hugging Face

```bash
# ~156G, 48 safetensor shards — use YOUR directory
hf download deepseek-ai/DeepSeek-V4-Flash-Vision-Exp \
  --local-dir ./DeepSeek-V4-Flash-Vision-Exp

du -sh ./DeepSeek-V4-Flash-Vision-Exp
ls ./DeepSeek-V4-Flash-Vision-Exp/model-*-of-00048.safetensors | wc -l   # expect 48
```

Copy that directory to **both** Spark nodes (or NFS-mount the same path on both). DSPARK needs **no** second download — draft weights load from this same checkpoint as `DeepseekV4ForCausalLMDSpark` (~5.5 GB).

### 2. Pull the SGLang image (both nodes)

```bash
docker pull lmsysorg/sglang:dev-v4f-2dgx-v2
```

### 3. Install the stream `load_weights` overlay

```bash
mkdir -p ./sglang-vision-exp/overlay/models
cp overlay/deepseek_v4_vl.py ./sglang-vision-exp/overlay/models/
cp scripts/run-v2-stream.sh ./sglang-vision-exp/
chmod +x ./sglang-vision-exp/run-v2-stream.sh
```

Set (or edit script defaults):

| Variable | Meaning |
| --- | --- |
| `MODEL_HOST` | Absolute path to your Vision-Exp directory on that node |
| `OVERLAY` | Directory containing `models/deepseek_v4_vl.py` |
| `DIST_ADDR` | Head RDMA/IP `:25000` (match your fabric) |
| NCCL ifaces | Match your ConnectX / NIC names if different from script defaults |

### 4. Launch TP=2

```bash
./sglang-vision-exp/run-v2-stream.sh 1   # worker, node-rank 1
./sglang-vision-exp/run-v2-stream.sh 0   # head, node-rank 0 — API on :30000
```

Wait for `The server is fired up and ready to roll!` (~10–15+ min: target ~74–76 GB/rank, then DSpark, then CUDA graphs).

### 5. Smoke

```bash
curl -s http://HEAD:30000/v1/models

curl -s http://HEAD:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"/models/DeepSeek-V4-Flash-Vision-Exp","messages":[{"role":"user","content":"Say hi"}],"max_tokens":32,"chat_template_kwargs":{"thinking":false}}'

curl -s http://HEAD:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"/models/DeepSeek-V4-Flash-Vision-Exp","messages":[{"role":"user","content":[
      {"type":"image_url","image_url":{"url":"https://raw.githubusercontent.com/sgl-project/sglang/main/examples/assets/example_image.png"}},
      {"type":"text","text":"Describe this image in one short sentence."}
    ]}],"max_tokens":128,"chat_template_kwargs":{"thinking":false}}'
```

Container model id is `/models/DeepSeek-V4-Flash-Vision-Exp` (docker bind of `MODEL_HOST`).

### Hard requirements / gotchas

- Image **`lmsysorg/sglang:dev-v4f-2dgx-v2`** + **stream overlay** (required on GB10)
- `--disable-shared-experts-fusion`, `--weight-loader-drop-cache-after-load`
- Used here: `--speculative-algorithm DSPARK`, `--tool-call-parser deepseekv4`, `--reasoning-parser deepseek-v4`
- **Never** `--weight-loader-disable-mmap` on GB10
- Non-v2 image cannot load Vision (`aligner` missing); thin VL overlays on non-v2 fail at runtime (`HashTopK` / `input_ids is None`)
- Clients should attach images as multimodal parts; a bare filesystem path is text and encourages a `read` tool call

Tear down is just removing the containers; image + weights + overlay stay on disk. Relaunch with the same `run-v2-stream.sh` commands.

---

## Official cookbook (what LMSYS says)

DGX Spark row (Flash Official FP4, NVFP4, **Flash Vision FP4**):

- Image: **`lmsysorg/sglang:dev-v4f-2dgx-v2`** only (branch `b12x-vision` @ `452239a74f`)
- Env: `SGLANG_SM120_FLASHMLA_BACKEND=b12x`, `SGLANG_B12X_MAX_TOKENS` == `--chunked-prefill-size`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- Vision: same flags as Official; native `image_url` on `/v1/chat/completions`
- Vision notes: engine auto-disables shared-experts fusion (HashTopK); DSpark accept ~3.2 vs ~3.9 on text Official
- Non-Spark preview `lmsysorg/sglang:dev-dsv4-flash-vision` is **not** the Spark path

Older `lmsysorg/sglang:dev-v4f-2dgx` (non-v2): text Flash-0731 OK, **no Vision** / missing `aligner`.

---

## Experiment timeline (what we tried)

### A. Stock `-v2` → OOM

Multi-thread load hits 48/48 shards → MemAvailable collapses → rank 0 scheduler **SIGKILL (-9)** before `Load weight end`. Lower context / mem-fraction, no DSPARK, no graphs, fusion off — still OOM. Same pattern for Vision and for 0731 on `-v2` without the stream patch.

### B. `--weight-loader-disable-mmap` → disaster

Forced CPU copies on UMA → host hang / hard reboot. **Never** on GB10.

### C. Non-v2 image for 0731 → text OK, Vision fail

`dev-v4f-2dgx` + fusion-off + drop-cache + DSPARK: Flash-0731 healthy (~72.9 GB + ~5.5 GB DSpark). Vision: `KeyError: aligner.gate_up_proj.weight`.

### D. Thin VL overlay on non-v2 → load OK, runtime fail

Streaming load worked (~75 GB); decode hit `HashTopK` → `input_ids is None`. Abandoned. Reference: `scripts/run-thin-overlay.sh`.

### E. Full `-v2` module overlay onto non-v2 → ImportError

Missing `attn_cp_overlap_all_gather_into_tensor` / `get_attn_cp_overlap_group`. Need whole `-v2` runtime.

### F. Streaming patch on official `-v2` → **working**

Bind-mount only:

```text
overlay/deepseek_v4_vl.py
  → /sgl-workspace/sglang/python/sglang/srt/models/deepseek_v4_vl.py
```

Target load end ≈ **74.5–76 GB**; with DSPARK draft ≈ **5.7 GB**; CUDA graphs OK.

| Case | TTFT | Decode tok/s | Notes |
| --- | --- | --- | --- |
| text-decode-256 | ~0.28 s | **~30.5** | thinking off |
| image-describe | ~0.47 s | **~33** | `image_tokens` ~155 |

First-bench DSpark accept ~2.0–2.4; later live traffic ~2.7–3.75 (cookbook Vision ~3.2).

### G. Disk pressure

Keep tens of GB free on Spark root (docker images + mmap). Prune old tags when root nears full.

### H. Tool / reasoning parsers

Without parsers, DeepSeek DSML tool markup can leak as assistant text (vLLM often hid this). Enable:

```text
--tool-call-parser deepseekv4
--reasoning-parser deepseek-v4
```

---

## Cookbook vs working command (deltas)

**Same intent:** `-v2` image, TP=2 / nnodes=2 / RoCE docker flags, `b12x` env + `--moe-runner-backend b12x`, DSPARK, chunked-prefill 8192, `mem-fraction-static 0.80`, long context.

**Extra / changed:**

1. Bind-mount streamed `deepseek_v4_vl.py` (**required**)
2. `--disable-shared-experts-fusion`
3. `--weight-loader-drop-cache-after-load`
4. `--tool-call-parser deepseekv4`
5. `--reasoning-parser deepseek-v4`
6. `cuda-graph-max-bs-decode 32`, `max-running-requests 32`
7. Offline HF envs when weights are local (`HF_HUB_OFFLINE=1`)

**Not used:** stock VL without stream; `--weight-loader-disable-mmap`; non-v2 as Vision host; thin VL overlay on non-v2 for production.

Repo files:

- `scripts/run-v2-stream.sh` — working launch
- `overlay/deepseek_v4_vl.py` — streaming `load_weights` patch
- `scripts/run-thin-overlay.sh` — failed experiment (reference)

---

## Patch detail (`load_weights`)

Stock (bad on GB10):

```python
llm_weights = []
for name, loaded_weight in weights:
    if vision / aligner / sentinel:
        load locally
    else:
        llm_weights.append((name, loaded_weight))  # holds entire LLM
self.language_model.load_weights(llm_weights)
```

Fixed (stream):

```python
def _iter_llm_weights():
    for name, loaded_weight in weights:
        if vision / aligner / sentinel:
            load locally
        else:
            yield name, loaded_weight
self.language_model.load_weights(_iter_llm_weights())
```

Upstream: worth filing against `sgl-project/sglang` / Spark Vision recipe.

---

## Open / follow-ups

- [ ] Confirm clients get structured `tool_calls` (not raw DSML) with parsers on
- [ ] Upstream PR: stream `deepseek_v4_vl.load_weights`
- [ ] Keep Spark root disk from filling (prune old images)

---

## One-line answer to “a lot of changes vs recipe?”

**Almost the same stack as the cookbook — plus one mandatory VL weight-loader stream patch, fusion-off + drop-cache, and DeepSeek tool/reasoning parsers.** Without the stream patch, stock `-v2` never finishes weight load on our 2× GB10 UMA pair.
