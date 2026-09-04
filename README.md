# DeepSeek-V4-Flash-Vision-Exp on 2× DGX Spark (SGLang)

**Date:** 2026-09-04  
**Last updated:** 2026-09-04 12:42 PT  
**Hosts:** `gx10-30c1` (`192.168.1.68`) + `gx10-3fe8` (`192.168.1.67`)  
**API:** `http://192.168.1.68:30000`  
**Cookbook:** https://docs.sglang.io/cookbook/autoregressive/DeepSeek/DeepSeek-V4.md

## Current status (2026-09-04 09:40 PT)

**UP.** Both ranks running ~31 min. `/v1/models` HTTP 200. Live traffic from M5 (`192.168.1.57`) returning 200.

| Piece | Value |
| --- | --- |
| Target load | 74.0 GB (rank0) / 75.0 GB (rank1) |
| DSpark draft | 5.52–5.53 GB (`DeepseekV4ForCausalLMDSpark`) |
| READY | 09:19 PT (`The server is fired up and ready to roll!`) |
| Parsers | `--tool-call-parser deepseekv4` + `--reasoning-parser deepseek-v4` |
| Live decode (sample) | accept len **2.7–3.75**, gen **~37–44 tok/s**, `cuda graph: True` |

This note dumps the full experiment path: official SGLang recipe → what broke on our GB10 UMA pair → the one-file fix → working launch → deltas vs cookbook → Pi wiring → open issues.

---

## TL;DR

| Item | Cookbook / stock | What actually worked here |
| --- | --- | --- |
| Image | `lmsysorg/sglang:dev-v4f-2dgx-v2` | Same image **+ bind-mount** streamed `deepseek_v4_vl.py` |
| Weight load | Stock VL `load_weights` | **Must stream** — stock buffers all LLM tensors → ~2× peak RSS → OOM on GB10 |
| Shared-experts fusion | Engine auto-disables on Vision; recipe may still show fusion path | Explicit `--disable-shared-experts-fusion` |
| Page cache | Optional | `--weight-loader-drop-cache-after-load` (important on UMA) |
| Speculative | DSPARK on balanced Spark cell | DSPARK works **after** streaming patch (draft is `DeepseekV4ForCausalLMDSpark` ~5.7 GB, not a 2nd full VL) |
| Decode CUDA graphs | On | Works on `-v2` with DSPARK (verify graphs). Failed earlier on non-v2 thin overlay (`HashTopK` + `input_ids=None`) |
| Tool / reasoning parsers | Playground toggles | Added `--tool-call-parser deepseekv4` + `--reasoning-parser deepseek-v4` (Pi was dumping raw DSML; vLLM hid this) |
| Never use on GB10 | — | `--weight-loader-disable-mmap` (host hang / hard reboot) |

**Root cause of stock `-v2` OOM:**  
`deepseek_v4_vl.DeepseekV4ForCausalLM.load_weights` collected every non-vision tensor into a Python `list` before calling `language_model.load_weights`. On 128 GB UMA that peaks near **2×** weight footprint and the OS OOM-kills the scheduler (exit -9) after 48/48 shards, never reaching `Load weight end`.

**Fix:** stream via a generator (see `overlay/deepseek_v4_vl.py`). After that, load end ≈ **74–76 GB/rank**, same ballpark as Flash-0731 text.

---

## Reproduce (external)

Share this section with others. **Do not** depend on Anemll lab paths (`/Volumes/TB36/...`, `/home/anemll/...`, M3U, or LAN IPs).

### 1. Download weights from Hugging Face

Repo: [`deepseek-ai/DeepSeek-V4-Flash-Vision-Exp`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp)

```bash
# ~156G, 48 safetensor shards — pick YOUR local directory
hf download deepseek-ai/DeepSeek-V4-Flash-Vision-Exp \
  --local-dir ./DeepSeek-V4-Flash-Vision-Exp

# integrity
du -sh ./DeepSeek-V4-Flash-Vision-Exp
ls ./DeepSeek-V4-Flash-Vision-Exp/model-*-of-00048.safetensors | wc -l   # expect 48
```

Copy that directory to **both** Spark nodes (or NFS-mount the same path on both). DSPARK needs **no** second download — draft weights load from this same checkpoint as `DeepseekV4ForCausalLMDSpark` (~5.5 GB).

### 2. Pull the SGLang image (both nodes)

```bash
docker pull lmsysorg/sglang:dev-v4f-2dgx-v2
```

### 3. Install the stream `load_weights` overlay

Stock VL loader OOMs on GB10 UMA (buffers the full LLM in a Python list). Use this repo’s patch:

```bash
# on each node (or shared filesystem)
mkdir -p /path/to/sglang-vision-exp/overlay/models
cp overlay/deepseek_v4_vl.py /path/to/sglang-vision-exp/overlay/models/
cp scripts/run-v2-stream.sh /path/to/sglang-vision-exp/
chmod +x /path/to/sglang-vision-exp/run-v2-stream.sh
```

Edit `run-v2-stream.sh` (or export env vars) so:

| Variable | Meaning |
| --- | --- |
| `MODEL_HOST` | Absolute path to **your** Vision-Exp directory on that node |
| `OVERLAY` | Directory that contains `models/deepseek_v4_vl.py` |
| `DIST_ADDR` | Your head RDMA/IP `:25000` (recipe uses RoCE; match your fabric) |
| Image / NCCL ifaces | Match your ConnectX / NIC names if they differ from the script defaults |

### 4. Launch TP=2

```bash
# worker then head (or both quickly)
./run-v2-stream.sh 1   # on worker, node-rank 1
./run-v2-stream.sh 0   # on head, node-rank 0 — API on :30000
```

Wait for `The server is fired up and ready to roll!` (~10–15+ min: target ~74–76 GB/rank, then DSpark, then CUDA graphs).

### 5. Smoke

```bash
curl -s http://HEAD:30000/v1/models

curl -s http://HEAD:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"/models/DeepSeek-V4-Flash-Vision-Exp","messages":[{"role":"user","content":"Say hi"}],"max_tokens":32,"chat_template_kwargs":{"thinking":false}}'
```

Model id inside the container is `/models/DeepSeek-V4-Flash-Vision-Exp` (docker bind of `MODEL_HOST`).

### Hard requirements / gotchas

- Image **`lmsysorg/sglang:dev-v4f-2dgx-v2`** + **stream overlay** (required on GB10)
- `--disable-shared-experts-fusion`, `--weight-loader-drop-cache-after-load`
- Optional but used here: `--speculative-algorithm DSPARK`, `--tool-call-parser deepseekv4`, `--reasoning-parser deepseek-v4`
- **Never** `--weight-loader-disable-mmap` on GB10 (host hang / hard reboot)
- Non-v2 image cannot load Vision (`aligner` missing); thin VL overlays on non-v2 fail at runtime

Cookbook: https://docs.sglang.io/cookbook/autoregressive/DeepSeek/DeepSeek-V4.md

Lab-only layout (Anemll TB36 / Spark absolute paths) lives in [Model inventory](#model-inventory-tb36--sparks) and [Redeploy later](#redeploy-later-working-path) below — skip those when sharing.

---

## Hardware / network

| Role | Hostname | LAN | RDMA (ConnectX-7) |
| --- | --- | --- | --- |
| Head (rank 0, API) | `gx10-30c1` | `192.168.1.68` | `10.200.0.1` (`enp1s0f0np0`) |
| Worker (rank 1) | `gx10-3fe8` | `192.168.1.67` | peer on `10.200.0.0/31` |

- TP=2 across two nodes; `--dist-init-addr 10.200.0.1:25000`
- Docker needs: `--network host --ulimit memlock=-1 --cap-add IPC_LOCK --device /dev/infiniband`
- NCCL envs in launch script pin IB HCAs / RoCE / socket ifaces (see `scripts/run-v2-stream.sh`)

**Weights:** see [Model inventory](#model-inventory-tb36--sparks) below. Serving path on each Spark:

```text
/home/anemll/models/DeepSeek-V4-Flash-Vision-Exp
```

Canonical archive: TB36 (TrueNAS SMB, usually mounted on **M3U** as `/Volumes/TB36`).

---

## Official cookbook (what LMSYS says)

DGX Spark row (all three cells: Flash Official FP4, NVFP4, **Flash Vision FP4**):

- Image: **`lmsysorg/sglang:dev-v4f-2dgx-v2`** only (branch `b12x-vision` @ `452239a74f`) — SM12x `b12x` MoE + compressed-MLA + Vision (#37253) + image-prefill fix
- Env (recipe): `SGLANG_SM120_FLASHMLA_BACKEND=b12x`, `SGLANG_B12X_MAX_TOKENS` == `--chunked-prefill-size`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- Vision: same flags as Official; native `image_url` on `/v1/chat/completions`
- Vision notes: engine auto-disables shared-experts fusion (HashTopK); DSpark accept ~3.2 vs ~3.9 on text Official
- Non-Spark Vision preview image `lmsysorg/sglang:dev-dsv4-flash-vision` is **not** the Spark path

We also had locally: `lmsysorg/sglang:dev-v4f-2dgx` (older, text-0731 OK, **no Vision** / missing `aligner`).

---

## Experiment timeline (what we tried)

### A. Stock `-v2` + cookbook-ish flags → OOM

Repeatedly:

1. Multi-thread load reaches **48/48 shards**
2. Host MemAvailable collapses
3. Rank 0 scheduler **SIGKILL (-9)** — never prints `Load weight end`
4. Tried: lower `context-length` / `mem-fraction-static`, no DSPARK, no cuda graphs, fusion off — **still OOM** on `-v2`
5. Same OOM pattern on Vision **and** on 0731 when forced onto `-v2` without the stream patch

### B. `--weight-loader-disable-mmap` → disaster

Forced CPU copies on UMA → head Spark hung, SSH dead, **manual reboot**. Lesson: **never** disable mmap on GB10 for this model.

### C. Non-v2 image (`dev-v4f-2dgx`) for 0731 → success (text only)

Proven recipe for text Flash-0731:

- Image: `lmsysorg/sglang:dev-v4f-2dgx`
- `--disable-shared-experts-fusion`
- `--weight-loader-drop-cache-after-load`
- DSPARK + cookbook env
- `Load weight end` ≈ **72.9 GB** target + **~5.5 GB** DSpark

Vision on this image: `KeyError: aligner.gate_up_proj.weight` (no VL path).

### D. Thin overlay: VL/ViT onto non-v2 → load OK, runtime fail

Mounted only:

- patched `deepseek_v4.py` (move `EntryClass` to VL; skip `bias_vl`; suffix-exact `.gate.bias` remap)
- `deepseek_v4_vl.py` / `deepseek_v4_vit.py` / multimodal processor

With **streaming** `load_weights`, weights loaded (~75 GB). Then:

- CUDA graph / eager decode: `HashTopK` → `input_ids is None` → `AttributeError`
- ABI mismatch: non-v2 MoE path ≠ what VL wrapper expects

**Abandoned for serving** (useful as a dead-end note). Script kept: `scripts/run-thin-overlay.sh`.

### E. Full `-v2` module overlay → ImportError

Tried bind-mounting v2 `dp_attention` / more modules onto non-v2 → missing symbols (`attn_cp_overlap_all_gather_into_tensor`, `get_attn_cp_overlap_group`). Need whole `-v2` runtime.

### F. Streaming patch on official `-v2` → **working**

Only mount:

```text
overlay/deepseek_v4_vl.py
  → /sgl-workspace/sglang/python/sglang/srt/models/deepseek_v4_vl.py
```

Then:

1. Target load end ≈ **74.5–76 GB**, avail ≈ **35–36 GB**
2. Without DSPARK: API up; text + image smoke HTTP 200
3. With DSPARK: draft `DeepseekV4ForCausalLMDSpark` ≈ **5.7 GB**; target+draft verify CUDA graphs capture OK; `"The server is fired up and ready to roll!"`
4. Speed (streaming client, thinking off), 2026-09-04:

| Case | TTFT | Decode tok/s | Notes |
| --- | --- | --- | --- |
| text-short | ~0.27 s | noisy (~50) | tiny outs |
| text-decode-256 | ~0.28 s | **~30.5** | steady |
| image-describe | ~0.47 s | **~33** | `image_tokens` ~155 |

First-bench decode logs: `cuda graph: True`, **accept len ~2.0–2.4**. After parsers relaunch + live Pi traffic (09:39 PT): **accept len ~2.7–3.75** (rate 0.34–0.55), gen **~37–44 tok/s** — in cookbook Vision ballpark (~3.2).

### G. Disk pressure

Head Spark hit **99%** root during debugging (many old vLLM images). Pruned ~40 GB reclaimable tags before reliable runs. Keep ≥ tens of GB free around weight mmap / docker.

### H. Pi (M5) client

`~/.pi/agent/models.json` provider `gx10-dspark`:

- `baseUrl`: `http://192.168.1.68:30000/v1`
- model id: `/models/DeepSeek-V4-Flash-Vision-Exp`
- `input`: `["text","image"]`
- default in `settings.json`

**Issue seen:** pasting a filesystem path made the model emit raw **DSML** `<｜DSML｜tool_calls>` instead of OpenAI `tool_calls` (vLLM previously parsed this). Mitigated by adding SGLang parsers (below). Also: Pi must attach images as multimodal parts, not only as a path string — path-as-text encourages tool-`read` behavior.

### I. Tool / reasoning parsers (Pi DSML leak)

Pi on M5 pasted a local image path. The model emitted raw DSML:

```text
<｜DSML｜tool_calls>
<｜DSML｜invoke name="read">
<｜DSML｜parameter name="path" string="true">/Users/anemll/Downloads/....png</｜DSML｜parameter>
</｜DSML｜invoke>
</｜DSML｜tool_calls>
```

vLLM used to parse this into OpenAI `tool_calls`; stock SGLang dumped it as assistant text. Cookbook Playground “Parsers” card is off by default on the Spark cell.

**Fix (in `run-v2-stream.sh`):**

```text
--tool-call-parser deepseekv4
--reasoning-parser deepseek-v4
```

Relaunched 2026-09-04 ~09:09 PT (API down ~10 min during target + draft + graph capture). Back READY 09:19 PT.

**Also:** Pi should attach images as multimodal parts (`input: ["text","image"]` is set). A bare filesystem path is still just text and encourages a `read` tool call even with parsers on.

---

## Cookbook vs our working command (delta list)

**Same as cookbook intent:**

- Image `dev-v4f-2dgx-v2`
- TP=2 / nnodes=2 / RoCE docker flags
- `b12x` env + `--moe-runner-backend b12x`
- `--speculative-algorithm DSPARK`
- `chunked-prefill-size` / `SGLANG_B12X_MAX_TOKENS` = 8192
- `mem-fraction-static 0.80`, long `context-length`

**Extra / changed (our deltas):**

1. **Bind-mount streamed `deepseek_v4_vl.py`** (required — not in cookbook)
2. **`--disable-shared-experts-fusion`** (explicit; Vision should auto-disable anyway)
3. **`--weight-loader-drop-cache-after-load`**
4. **`--tool-call-parser deepseekv4`**
5. **`--reasoning-parser deepseek-v4`**
6. Conservative decode graph / concurrency initially; now `cuda-graph-max-bs-decode 32`, `max-running-requests 32`
7. Offline HF envs (`HF_HUB_OFFLINE=1`) — weights local

**Not used (failed or harmful):**

- Stock VL without stream patch
- `--weight-loader-disable-mmap`
- Non-v2 as Vision host (missing aligner / HashTopK path)
- Thin VL overlay on non-v2 for production

---

## How to launch (working path)

On **both** Sparks (scripts live on Sparks under `/home/anemll/sglang-vision-exp/`; copies archived here):

```bash
# head
/home/anemll/sglang-vision-exp/run-v2-stream.sh 0
# worker
/home/anemll/sglang-vision-exp/run-v2-stream.sh 1
```

Archived copies:

- `scripts/run-v2-stream.sh` — production launch
- `overlay/deepseek_v4_vl.py` — streaming `load_weights` patch
- `scripts/run-thin-overlay.sh` — failed non-v2 experiment (reference only)

Smoke:

```bash
# text
curl -s http://192.168.1.68:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"/models/DeepSeek-V4-Flash-Vision-Exp","messages":[{"role":"user","content":"Say hi"}],"max_tokens":32,"chat_template_kwargs":{"thinking":false}}'

# image
curl -s http://192.168.1.68:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"/models/DeepSeek-V4-Flash-Vision-Exp","messages":[{"role":"user","content":[
      {"type":"image_url","image_url":{"url":"https://raw.githubusercontent.com/sgl-project/sglang/main/examples/assets/example_image.png"}},
      {"type":"text","text":"Describe this image in one short sentence."}
    ]}],"max_tokens":128,"chat_template_kwargs":{"thinking":false}}'
```

Expect model id:

```text
/models/DeepSeek-V4-Flash-Vision-Exp
```

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

Upstream: worth filing against `sgl-project/sglang` / Spark Vision recipe — cookbook claims verified on 2× Spark, but stock VL loader OOMs on our GB10 UMA unless streamed.

---

## Open / follow-ups

- [x] Parsers live on serving stack (`deepseekv4` / `deepseek-v4`); API READY 09:19 PT
- [ ] Confirm in Pi that DSML now lands as structured `tool_calls` (not raw text) — server-side flags are on; client-side image attach still required
- [x] DSpark accept length: live samples **~2.7–3.75** (was ~2.0–2.4 on first bench)
- [ ] Upstream PR: stream `deepseek_v4_vl.load_weights`
- [ ] Keep Spark root disk from filling (prune old container images)

---

## Model inventory (TB36 + Sparks)

> **Lab-only.** External reproduce uses the HF download above, not these paths.

Verified **2026-09-04 12:40 PT** from M3U (`192.168.1.38`). TB36 is the TrueNAS SMB share `//streambox@truenas…/TB36` mounted at `/Volumes/TB36` (not on M5 unless you mount it there).

### Required for this Vision serve

| Role | HF / name | TB36 (canonical) | Sparks (serving) | Size / shards |
| --- | --- | --- | --- | --- |
| Target + Vision + DSpark draft | `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp` | `/Volumes/TB36/Models/DS/DeepSeek-V4-Flash-Vision-Exp` | `/home/anemll/models/DeepSeek-V4-Flash-Vision-Exp` on **both** `.68` and `.67` | **156–157G**, **48/48** `model-*-of-00048.safetensors` |

DSPARK does **not** need a second download. With `--speculative-algorithm DSPARK`, SGLang loads draft weights as `DeepseekV4ForCausalLMDSpark` (~5.5 GB) from the **same** Vision-Exp checkpoint.

Container image (also required to redeploy):

| Image | Where |
| --- | --- |
| `lmsysorg/sglang:dev-v4f-2dgx-v2` | Pulled on both Sparks (~33 GB) |

Runtime patch (bind-mount, not a model):

| Asset | Live on Sparks | Archived in this folder |
| --- | --- | --- |
| Streamed `load_weights` | `/home/anemll/sglang-vision-exp/overlay-v2-stream/models/deepseek_v4_vl.py` | `overlay/deepseek_v4_vl.py` |
| Launch script | `/home/anemll/sglang-vision-exp/run-v2-stream.sh` | `scripts/run-v2-stream.sh` |

### Related (used in bring-up, not required for Vision serve)

| Name | TB36 | Sparks | Notes |
| --- | --- | --- | --- |
| `DeepSeek-V4-Flash-0731` | `/Volumes/TB36/Models/DS/DeepSeek-V4-Flash-0731` (155G, 48 shards) | `/home/anemll/models/DeepSeek-V4-Flash-0731` (156G) | Text baseline on non-v2 image during experiments |
| NVFP4 Flash variants | `/Volumes/TB36/Models/DS/DeepSeek-V4-Flash-NVFP4*` | symlink / prepared dirs on Sparks | **Not** used by this SGLang Vision recipe |

### Quick integrity check

```bash
# TB36 (from M3U)
V=/Volumes/TB36/Models/DS/DeepSeek-V4-Flash-Vision-Exp
du -sh "$V"
ls "$V"/model-*-of-00048.safetensors | wc -l   # expect 48

# Sparks
for h in 192.168.1.68 192.168.1.67; do
  ssh anemll@$h 'du -sh ~/models/DeepSeek-V4-Flash-Vision-Exp; ls ~/models/DeepSeek-V4-Flash-Vision-Exp/model-*-of-00048.safetensors | wc -l'
done
```

---

## Redeploy later (working path)

> **Lab-only** (Anemll Sparks / TB36). For sharing, use [Reproduce (external)](#reproduce-external).

Assumes weights already on both Sparks (or re-rsync from TB36 first).

### 0. Prerequisites

- Head `gx10-30c1` `192.168.1.68`, worker `gx10-3fe8` `192.168.1.67`
- RoCE up: `10.200.0.1` / `enp1s0f0np0`, dist addr `10.200.0.1:25000`
- Image present: `docker images | grep dev-v4f-2dgx-v2` (else `docker pull lmsysorg/sglang:dev-v4f-2dgx-v2` on **both**)
- Overlay + script under `/home/anemll/sglang-vision-exp/` (restore from this repo’s `overlay/` + `scripts/run-v2-stream.sh` if missing)
- Free disk on Sparks (keep tens of GB free; head previously hit 99%)
- **Never** pass `--weight-loader-disable-mmap`

### 1. (If needed) copy weights from TB36 → Sparks

From **M3U** (where TB36 is mounted):

```bash
SRC=/Volumes/TB36/Models/DS/DeepSeek-V4-Flash-Vision-Exp/
# confirm complete first
ls "$SRC"/model-*-of-00048.safetensors | wc -l   # 48

rsync -aH --info=progress2 "$SRC" anemll@192.168.1.68:~/models/DeepSeek-V4-Flash-Vision-Exp/
rsync -aH --info=progress2 "$SRC" anemll@192.168.1.67:~/models/DeepSeek-V4-Flash-Vision-Exp/
```

### 2. Stop anything holding the GPUs

```bash
ssh anemll@192.168.1.68 'docker rm -f sglang-vision-rank0 2>/dev/null; docker ps -a'
ssh anemll@192.168.1.67 'docker rm -f sglang-vision-rank1 2>/dev/null; docker ps -a'
# also stop old vLLM :8888 stacks if present
```

### 3. Launch TP=2 (worker first or both quickly)

```bash
ssh anemll@192.168.1.67 '/home/anemll/sglang-vision-exp/run-v2-stream.sh 1'
ssh anemll@192.168.1.68 '/home/anemll/sglang-vision-exp/run-v2-stream.sh 0'
```

Expect ~10–15+ min to READY (target load ~74–76 GB/rank, then DSpark ~5.5 GB, then CUDA graph capture). Watch:

```bash
ssh anemll@192.168.1.68 'docker logs -f sglang-vision-rank0'   # look for "fired up and ready to roll"
curl -sS http://192.168.1.68:30000/v1/models
```

### 4. Smoke

```bash
# text
curl -s http://192.168.1.68:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"/models/DeepSeek-V4-Flash-Vision-Exp","messages":[{"role":"user","content":"Say hi"}],"max_tokens":32,"chat_template_kwargs":{"thinking":false}}'

# image (native multimodal)
curl -s http://192.168.1.68:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"/models/DeepSeek-V4-Flash-Vision-Exp","messages":[{"role":"user","content":[
      {"type":"image_url","image_url":{"url":"https://raw.githubusercontent.com/sgl-project/sglang/main/examples/assets/example_image.png"}},
      {"type":"text","text":"Describe this image in one short sentence."}
    ]}],"max_tokens":128,"chat_template_kwargs":{"thinking":false}}'
```

### 5. Pi (M5) client

`~/.pi/agent/models.json` provider `gx10-dspark`:

- `baseUrl`: `http://192.168.1.68:30000/v1`
- model: `/models/DeepSeek-V4-Flash-Vision-Exp`
- `input`: `["text","image"]`

Attach images as multimodal parts (not a bare filesystem path). Parsers are already in `run-v2-stream.sh`.

### 6. Tear down

```bash
ssh anemll@192.168.1.68 'docker rm -f sglang-vision-rank0'
ssh anemll@192.168.1.67 'docker rm -f sglang-vision-rank1'
```

Weights on TB36/Sparks can stay; only containers need removing to free GPUs for another backend.

---

## Related paths

| What | Where |
| --- | --- |
| TB36 share | TrueNAS SMB → `/Volumes/TB36` (typically on **M3U**) |
| Vision weights (archive) | `/Volumes/TB36/Models/DS/DeepSeek-V4-Flash-Vision-Exp` |
| Vision weights (serve) | `/home/anemll/models/DeepSeek-V4-Flash-Vision-Exp` on both Sparks |
| Live scripts / overlay | `/home/anemll/sglang-vision-exp/` on both Sparks |
| This write-up | `ML_playground/SGLang-DSv4F-vision-2xSparks/` (M5; path also reachable when M3U has the same tree) |
| Pi models | `~/.pi/agent/models.json` on M5 (`gx10-dspark` → `:30000`) |
| Prior vLLM DSpark stack | stopped/removed to free GPUs for Vision (`:8888`) |

---

## One-line answer to “a lot of changes vs recipe?”

**Almost the same stack as the cookbook — plus one mandatory VL weight-loader stream patch, fusion-off + drop-cache, and DeepSeek tool/reasoning parsers for Pi.** Without the stream patch, stock `-v2` never finishes weight load on our 2× GB10 UMA pair.
