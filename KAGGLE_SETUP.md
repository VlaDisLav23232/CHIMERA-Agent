# Running the baseline on Kaggle (with Ollama)

**Goal:** run the unchanged baseline agent on Task 1 cases on a free Kaggle T4 and compare the answers with the reference.

**Why Ollama and not vLLM:** tested on 17 Oct. **vLLM can't run Gemma 4 on a T4.**
- Gemma 4 has attention heads of size 512.
- vLLM's Triton attention kernel then needs 96 KB of shared memory, but the T4 only has 64 KB (`OutOfResources: shared memory`).
- Ollama (llama.cpp underneath) has its own kernels that work on a T4.
- The baseline supports any OpenAI-compatible server via `model.provider=openai`, so the agent code stays unchanged.
- For fair comparisons, the baseline and our agent must both run on the **same Ollama model tag**.

**Model:** `gemma4:e2b-it-q8_0`, the same Gemma 4 E2B as the official baseline, in 8-bit (6.0 GB). The difference from full precision is minimal. The official baseline uses bf16 via vLLM on an A10G, so our numbers are close to official but not identical. Say so in the report.

Expected time: about 30–40 minutes, most of it installing and downloading.

---

## Part A — one-time setup (outside Kaggle)

### A1. Hugging Face (for the guideline-search model only)
1. https://huggingface.co/google/embeddinggemma-300m: **gated, Gemma licence.** Open it **while logged in** and click **"Acknowledge license"** at the top of the model card. Access is granted immediately.
2. Create a **read** token: Settings → Access Tokens → New token (type *Read*). Copy it.

The language model itself comes from Ollama, not Hugging Face.

### A2. Kaggle account
1. Settings → **Phone verification**. Without it there's no GPU and no internet in notebooks.
2. **One account per person.** Kaggle forbids multiple accounts.

### A3. Prepare the data and upload it as a private dataset

**Our local release** (`data/`, gitignored) is laid out differently from what the baseline expects:
```
data/task1/PT-pseudo_<id>/          195 cases, no agent_input/ level
    structured-prompt.json                                (all 195)
    prostate-biopsy-decision-clinical-data.json           (all 195)
    prostate-modality-level-neural-representations.json   (191; 4 cases have none, which is fine because the image predictor is off)
    prostate-biopsy-decision.json                ← LABEL  (91 cases)
    prostate-biopsy-decision-reasoning.json      ← LABEL  (91 cases)
```
- The baseline looks for cases in `<data_root>/task1/agent_input/<case>/`.
- Our labels sit **inside the case folders**, next to the inputs.
- The baseline's tools only read `structured-prompt.json` and `*-clinical-data.json`, so nothing leaks today. But our own code must never be able to see the labels, so we separate them.

**Steps:**
1. Run this in the repo root on your laptop:
   ```
   python scripts/prepare_kaggle_data.py
   ```
   It creates `kaggle_upload.zip` (~2.6 MB, gitignored) with this layout:
   ```
   task1/agent_input/<case>/   3 input files × 195 cases   ← the agent reads only this
   task1/labels/<case>/        2 label files × 91 cases    ← evaluation only
   ```
   Tasks 2 and 3 are left out on purpose. We don't need them, and it's less patient data to upload.
2. kaggle.com → **Create → New Dataset** → upload `kaggle_upload.zip` (Kaggle unpacks it).
3. **Visibility: Private.** This is real patient data under a non-commercial licence, so never make it public.
4. Dataset → Settings → **Collaborators**: add Oleksii.

---

## Part B — create the notebook

1. kaggle.com → **Create → New Notebook**
2. Right panel → **Session options**:
   - **Accelerator: GPU T4 x2** (Ollama uses one GPU for a model this size; the second doesn't hurt)
   - **Internet: On**
3. Right panel → **Add Input** → your private dataset.
4. **Add-ons → Secrets** → add `HF_TOKEN` = your Hugging Face token, and tick it for this notebook.
5. Run cells **interactively** (Shift+Enter) the first time, not "Save & Run All". A background run shows no log until it finishes.

---

## Part C — notebook cells

### Cell 1 — check the GPU and find the data path
```python
!nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
!find /kaggle/input -maxdepth 6 -type d -name task1
```
The second line prints something like `/kaggle/input/datasets/<user>/<dataset>/task1`. Put the part **before** `/task1` into `DATA_ROOT` in cell 2.

### Cell 2 — secrets and paths
```python
import os
from kaggle_secrets import UserSecretsClient
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")

DATA_ROOT    = "/kaggle/input/datasets/vladyslavsh0/chimera-agent-challenge-task-1"   # from cell 1; Oleksii's path differs
LABELS       = f"{DATA_ROOT}/task1/labels"      # never passed to the agent
OUT_DIR      = "/kaggle/working/output"
REPO         = "/kaggle/working/chimera-agent-baseline"
OLLAMA_MODEL = "gemma4:e2b-it-q8_0"
```

### Cell 3 — clone and install the baseline (~2–3 min, no vLLM)
```python
!git clone --depth 1 https://github.com/DIAGNijmegen/chimera-agent-baseline.git {REPO}
%cd {REPO}
!pip install -q -e .
!pip install -q "transformers==5.12.1"   # pin from the baseline's requirements.txt
```
Red `ERROR: pip's dependency resolver ...` lines about protobuf, numba, google-*, gradio etc. are Kaggle's preinstalled packages that we don't use. **Ignore them.**

### Cell 4 — download the guideline-search model (~1.2 GB)
```python
!python scripts/download_embedding_model.py
!ls resources/embedding_model | head -3
```

### Cell 5 — install and start Ollama
```python
!apt-get install -y -qq zstd > /dev/null
!curl -fsSL https://ollama.com/install.sh | sh
import subprocess, time
ollama_env = {**os.environ,
              "OLLAMA_HOST": "127.0.0.1:11434",
              "OLLAMA_CONTEXT_LENGTH": "16384"}   # Ollama's default context is far too short for the agent
ollama_proc = subprocess.Popen(["ollama", "serve"], env=ollama_env,
                               stdout=open("/kaggle/working/ollama.log", "w"), stderr=subprocess.STDOUT)
time.sleep(5)
!ollama --version
```
- **Gemma 4 tool calling needs Ollama ≥ 0.22.0.** Older versions produce poor tool calls. Check the version line.
- If the Kaggle session restarts, run cells 2 and 5 again: the Ollama server stops with the session.

### Cell 6 — download the language model (~6 GB, ~1–2 min)
```python
!ollama pull {OLLAMA_MODEL}
!ollama list
```

### Cell 7 — pick 5 labeled test cases
We choose **labeled** cases so we can compare with the reference. With `agent.limit=5` the baseline would take the first 5 folders alphabetically, which may have no labels.
```python
PIDS = sorted(os.listdir(LABELS))[:5]
PIDS_ARG = "[" + ",".join(PIDS) + "]"
print(PIDS)
```

### Cell 8 — run the baseline agent
```python
!python -m chimera_agent_baseline.run \
    'agent.tasks=[1]' 'agent.pids={PIDS_ARG}' \
    paths.data_root={DATA_ROOT} \
    paths.output_dir={OUT_DIR} \
    paths.embedding_model_dir=resources/embedding_model \
    model.provider=openai \
    model.base_url=http://127.0.0.1:11434/v1 \
    model.model_id={OLLAMA_MODEL} \
    2>&1 | tee /kaggle/working/run_5cases.log
```
- `paths.embedding_model_dir=...`: works around a bug in the baseline. The download script saves the model to `resources/embedding_model`, but the config looks in `model/embedding_model`. Without this override, **the guideline search is silently switched off.**
- `model.provider=openai` + `base_url`: the baseline talks to Ollama through its OpenAI-compatible API. vLLM isn't used at all.

### Cell 9 — check that Ollama used the GPU
```python
!grep -iE "offload|layers.*gpu|CUDA|compute" /kaggle/working/ollama.log | tail -8
!nvidia-smi --query-gpu=memory.used --format=csv
```
- **Good:** something like `offloaded 36/36 layers to GPU`, and several GB of GPU memory in use.
- **Bad:** 0 layers on the GPU means it ran on the CPU and is very slow (see Troubleshooting).

### Cell 10 — compare with the reference
```python
import json, glob
!grep -iE "embedding service|loaded .* tools|error|Traceback" /kaggle/working/run_5cases.log | head -40
for d in sorted(glob.glob(f"{OUT_DIR}/task1/*")):
    case = d.split('/')[-1]
    dec = json.load(open(f"{d}/prostate-biopsy-decision.json"))
    r = json.load(open(f"{d}/prostate-biopsy-decision-reasoning.json"))
    ref = json.load(open(f"{LABELS}/{case}/prostate-biopsy-decision.json"))
    ref_r = json.load(open(f"{LABELS}/{case}/prostate-biopsy-decision-reasoning.json"))
    print(f"\n=== {case}: agent={dec} ref={ref} {'OK' if dec == ref else 'WRONG'} | conf {r['confidence']} (ref {ref_r['confidence']})")
    print(f"reveal: {r['reveal_sequence']}")
    print(f"agent: {r['free_text'][:400]}")
    print(f"ref:   {ref_r['free_text']}")
```

**It works if:**
- [ ] The log shows `Embedding service ready`, not `No embedding model at ...`
- [ ] The log shows `loaded N tools from MCP server`
- [ ] All 5 cases have both JSON files
- [ ] `free_text` is readable English about the actual patient
- [ ] `reveal_sequence` isn't empty: the agent actually called tools

### Cell 11 — keep the outputs
```python
!cd /kaggle/working && zip -qr output_5cases.zip output run_5cases.log ollama.log && ls -lh output_5cases.zip
```
Download it from the right panel (Output → `/kaggle/working`).

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Connection refused` to `127.0.0.1:11434` | The Ollama server isn't running (e.g. after a session restart). Rerun cell 5, then check with `!curl -s 127.0.0.1:11434/api/version` |
| Ollama runs on the CPU (cell 9 shows 0 layers on the GPU) | Look at the end of `ollama.log` for the CUDA error. Restart the session with the GPU accelerator on, then rerun from cell 2 |
| Agent never calls tools / `reveal_sequence` empty | Check `ollama --version` ≥ 0.22.0 (older versions have broken Gemma 4 tool calls) |
| Context errors or truncated prompts | `OLLAMA_CONTEXT_LENGTH` wasn't set when the server started. Kill the server (`ollama_proc.kill()`) and rerun cell 5 |
| `401 / gated repo` (cell 4) | EmbeddingGemma licence not acknowledged, or the `HF_TOKEN` secret isn't attached to this notebook |
| `No cases ... found` | `DATA_ROOT` must point to the folder that **contains** `task1/` |
| `form_fill` fails after 3 retries, the whole run stops | Known baseline limitation (see below). Note the case ID, remove it from `PIDS`, rerun |

**Why not vLLM (tested 17 Oct, for the record):**
- **Install:** a plain `pip install vllm==0.25.0` installs the CUDA 13 build (`libcudart.so.13` error). The `+cu129` build from `wheels.vllm.ai` plus `transformers==5.12.1` gets it installed.
- **Loading:** the model loads fine.
- **Warm-up:** fails with `OutOfResources: shared memory, Required: 98304, Hardware limit: 65536`. That's a T4 hardware limit for Gemma 4's 512-wide attention heads. vLLM needs an A10G or L4 (rented GPU) for this model.

---

## Known baseline limitations (relevant for full runs later)
- **No resume.** A rerun processes every case again and overwrites the outputs.
- **One failed case stops the whole run.** `form_fill` raises after 3 failed validation attempts.
- **Workaround for full runs:** run in batches with `agent.pids` from our dev/test split. Later we add resume and per-case error handling in our own runner (PLAN task 2.2).
- **Sampling is random** (`temperature: 1.0`), so results vary between runs. For fair comparisons, fix the seed or lower the temperature in the same way for both systems.
