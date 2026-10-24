# Task 1 agent

Runs development cases through one of two systems:

- `baseline_scaffold`: the organizers' unchanged graph, as in the baseline run.
- `agent_v1`: our graph. It keeps the baseline's ReAct loop, tools, prompts and
  model, and replaces the final form-filling step. The decision step sees every
  retrieved document and guideline passage under an ID (`P`, `E1…`, `G1…`), writes
  2-4 claims that cite those IDs, and records citation, number and weight checks
  in a per-case trace.

Our code adds safe input preparation, explicit settings, readiness checks, a
per-case time limit, batch runs and run metadata.

Runs need Linux, macOS or WSL: the runner stops process groups with `os.killpg`
and the baseline's guideline search talks to its embedding service over a Unix
socket. Full runs are done on Kaggle with
[notebooks/02_kaggle_agent_run.ipynb](../notebooks/02_kaggle_agent_run.ipynb).

## Setup

Run from the project root. Reuse the Python 3.12 .venv:

    bash scripts/setup_agent.sh

This fetches baseline commit 1afa788efc21c0e0144706d75170b0a4cd267367 and installs
requirements-agent.txt, constrained by requirements.txt. To regenerate the lock:

    uv pip compile requirements-agent.in --python .venv/bin/python -o requirements-agent.txt

The upstream repository tracks bytecode and has a malformed Git LFS README.
Source verification disables the LFS filter for comparison and excludes
__pycache__ files and the downloaded model/embedding_model asset directory; changes to actual source still fail verification.

Install Ollama separately if needed, then start its local server in another terminal:

    OLLAMA_CONTEXT_LENGTH=16384 ollama serve

Accept the conditions at https://huggingface.co/google/embeddinggemma-300m.
Make a read token available as HF_TOKEN in your shell or ignored .env file.
Do not put tokens in scripts or send them through chat.

    bash scripts/setup_agent.sh --models

Downloads Gemma (~6 GB as reported by the baseline run) and EmbeddingGemma
(size depends on downloaded artifacts). Model setup is separate from case execution.
It never rebuilds the guideline corpus. No vLLM or cloud inference is needed.
The optional evaluator judge model is not part of scaffold setup.

## Run

    .venv/bin/python -m chimera_agent.run --name smoke_01 --system agent_v1

`--system` is required (`agent_v1` or `baseline_scaffold`). Without case options
the first sorted dev case from splits/task1.json runs. Repeat `--case <dev-case-id>`
for several cases, or use `--all-dev` for the whole dev split. Test IDs are rejected.
Use --config <json-path> for explicit settings; defaults are in configs/agent.json.
The endpoint must be local and the model must match the baseline.
Run names cannot be reused; choose a new name for a retry.

Each case runs in its own worker process. `wall_seconds` (at most 600) is the
limit per case and covers readiness, model warmup and case execution, excluding
downloads. A failed or timed-out case is recorded and the run moves on to the next
one. The run status is `succeeded` (all cases), `partial` or `failed` (none); the
command exits non-zero unless every case succeeded. A missing asset, failed
guideline query or wrong context setting fails the case explicitly. If an existing
Ollama server has another context setting, restart it yourself using the
documented command; the runner does not reconfigure it.

### WSL

On Windows, install a distribution (`wsl --install -d Ubuntu`), clone the repository
inside it (not under /mnt/c) and follow Setup there with Ollama installed inside WSL.
WSL gets half of the machine's memory by default; raise it with `memory=12GB` under
`[wsl2]` in `%UserProfile%\.wslconfig`, then `wsl --shutdown`.

## Outputs

All run files are local under results/<name>/ and may contain patient information:

- resources/: one index copy per run, so Chroma metadata writes leave the checkout unchanged.
- cases/<case>/input/: allowlisted copies of that case's inputs, with no reference answers.
- cases/<case>/worker.log: diagnostics.
- cases/<case>/work/: partial outputs kept if the case fails.
- output/task1/<case>/: the two official JSON files, published per case only on success.
- trace/<case>.json: agent_v1 only, the evidence items, tool calls, claims with
  citations, check results, timings and LLM call count (`schema: chimera_agent.trace/2`,
  defined in evidence.py).
- run_info.json: run and per-case status, effective settings, code/model identity and timing.

No output files are fabricated after an error. Staging is a filename allowlist,
not a security sandbox. Patient source files are treated as data.

After a run, evaluate it separately:

    .venv/bin/python -m evaluation.score results/agent_v1_dev --name agent_v1_dev --system agent_v1 --temperature 1.0

Scoring always covers the whole split: unrun cases count as failures. A run with
fewer cases checks file compatibility only and is not an agent performance result;
keep such runs out of the shared leaderboard.

## Architecture and checks

run.py owns the batch, staging, process supervision, metadata and success-only publication.
worker.py integrates the pinned baseline, manages the embedding service and runs
the chosen system. graph.py is the agent_v1 graph, decision.py its decision step,
evidence.py the evidence ledger, checks and trace schema. config.py and inputs.py
contain settings and input preparation. Inference does
not import the evaluator, which alone has access to reference answers.

    .venv/bin/python -m unittest discover -s tests -t .

Synthetic tests cover label isolation, wrong IDs, symlinks, settings, process
failure/timeouts, batch statuses, output publication, experiment identity, the
evidence ledger and checks, and the graph with a scripted model. A real run
additionally needs model access, enough memory and a working local server.
