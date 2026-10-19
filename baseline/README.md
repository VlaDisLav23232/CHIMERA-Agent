# How the baseline was run (18 Oct 2026)

A short walkthrough of how the baseline numbers in [results/baseline.md](../results/baseline.md) were produced, so you can reproduce them or rerun with our own agent.

**The notebook:** [chimera_baseline_full_run.ipynb](chimera_baseline_full_run.ipynb) has every cell I ran, in order. Upload it to Kaggle (**File → Import Notebook**), set `DATA_ROOT` in cell 2 to your path, and run it.

Explanations and troubleshooting for each cell are in two files:
- [KAGGLE_SETUP.md](../KAGGLE_SETUP.md): one-time setup and notebook cells 1–6.
- [RUN_ALL_CASES.md](../RUN_ALL_CASES.md): the runner, metrics and save cells (A–D).

**Result:** 91/91 labeled cases, accuracy 0.692, F1(yes) 0.800 (always "yes": 0.615 / 0.762). About 2 h of GPU time.

---

## 1. What's in `kaggle_upload/` (not in git)

The challenge data reorganised into the layout the baseline expects, with the labels split off. It's the content of the private Kaggle dataset `vladyslavsh0/chimera-agent-challenge-task-1`.

```
kaggle_upload/task1/
  agent_input/<case>/        195 cases: what the agent may read
      structured-prompt.json
      prostate-biopsy-decision-clinical-data.json
      prostate-modality-level-neural-representations.json   (missing for 4 cases, fine: image predictor is off)
  labels/<case>/             91 cases: reference answers, evaluation only
      prostate-biopsy-decision.json             "yes" / "no"
      prostate-biopsy-decision-reasoning.json
```

- Made from our local `data/task1/` release, where labels sit *inside* the case folders. They're split so the agent can never read them.
- Only Task 1. Tasks 2 and 3 are left out on purpose (less patient data uploaded).
- `kaggle_upload.zip` is the same folder zipped. That zip is what was uploaded to Kaggle.
- **Patient data: never commit it, never make the dataset public.** Both are gitignored.

**You don't need to rebuild it.** You're a collaborator on the dataset, so just add it as input in your notebook. Your `DATA_ROOT` path may differ from mine (cell 1 prints it).

---

## 2. Steps I followed

Each step matches a cell in the notebook.


1. **One-time setup** ([KAGGLE_SETUP.md](../KAGGLE_SETUP.md) Part A): Kaggle phone verification, accepted the Gemma licence for `google/embeddinggemma-300m` on Hugging Face, created a read token.
2. **Notebook** (Part B): new notebook with GPU T4 x2, Internet on, the dataset added, the `HF_TOKEN` secret attached, persistence off.
3. **Setup cells 1–6**: GPU check, paths (`REPO` in `/kaggle/temp`), clone + `pip install -e .` of the baseline, download the embedding model, start Ollama, pull `gemma4:e2b-it-q8_0`.
   - We use **Ollama, not vLLM**: vLLM can't run Gemma 4 on a T4 (shared memory error). The baseline talks to Ollama via `model.provider=openai`, so the agent code is unchanged.
4. **Cell A**: wrote `run_info.json` (baseline commit, Ollama version, model, GPU).
5. **Cell B (runner)**: `SCOPE = "labeled"`, `BATCH = 10`.
   - First a timing test with `MAX_BATCHES = 1`, then the full run with `MAX_BATCHES = None`, interactively.
   - 10 batches, ~13 min each. No case failed, so no retries.
6. **Cell D**: zipped outputs and logs into `baseline_run.zip` and downloaded it right away (`/kaggle/working` is lost when the session ends).
7. **Metrics (cell C)**: computed locally from the unpacked zip and `kaggle_upload/task1/labels`, written to [results/baseline.md](../results/baseline.md).

---

## 3. Where the outputs are

`baseline_run.zip` unpacked into `results/baseline/` (gitignored, contains patient details). Ask me for the zip if you need it.

```
results/baseline/
  run_info.json                     run settings
  output/task1/<case>/              91 × (decision + reasoning JSON)
  logs/batch_000..009.log           baseline log per batch
  ollama.log                        Ollama server log
  predictions_vs_reference.json     per-case ref / pred / bx (made locally)
```

---

## 4. Rerunning it (or running our agent the same way)

- Use the **same Ollama tag** (`gemma4:e2b-it-q8_0`) so comparisons are fair.
- Budget ~2 h of the 30 GPU-hours a week. Edit with Accelerator: None and switch to the GPU only to run.
- Runs longer than ~3 h: use **Save Version → Save & Run All** (background, up to 12 h). See "Step 3" in RUN_ALL_CASES.md.
- If a session dies halfway, the runner resumes: see "Continuing after a session ended halfway" in RUN_ALL_CASES.md.
