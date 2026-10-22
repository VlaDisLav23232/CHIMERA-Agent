# Local Task 1 scaffold

Runs one development case through the organizers' unchanged agent. Our code adds
safe input preparation, explicit local settings, readiness checks, a time limit
and run metadata. Graph changes are a later work item.

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

    .venv/bin/python -m chimera_agent.run --name scaffold_smoke

By default, selects the first sorted dev case from splits/task1.json.
Use --case <dev-case-id> to select another. Test IDs are rejected.
Use --config <json-path> for explicit settings; defaults are in configs/agent.json.
The endpoint must be local and the model must match the baseline.
Run names cannot be reused; choose a new name for a retry.

The 600-second limit covers readiness, model warmup and case execution, excluding
downloads. A missing asset, failed guideline query or wrong context setting
fails explicitly. If an existing Ollama server has another context setting,
restart it yourself using the documented command; the runner does not reconfigure it.

## Outputs

All run files are local under results/<name>/:

- resources/: per-run index copy, so Chroma metadata writes leave the checkout unchanged.
- input/: allowlisted copies of dev inputs, with no reference answers.
- output/task1/<case>/: two official JSON files, published only on success.
- run_info.json: status, effective settings, code/model identity and elapsed time.
- worker.log: diagnostics, potentially containing patient information.
- work/: partial outputs retained locally if a run fails.

No output files are fabricated after an error. Staging is a filename allowlist,
not a security sandbox. Patient source files are treated as data.

After a successful run, evaluate it separately:

    .venv/bin/python -m evaluation.score results/scaffold_smoke --name scaffold_smoke_dev --system baseline_scaffold --temperature 1.0

That scores all 60 dev cases: 59 unrun cases count as failures. Use this only to
check file compatibility; it is not an agent performance result. Avoid registering
partial smoke runs in the shared leaderboard. Use the full dev run later.

## Architecture and checks

run.py owns staging, process supervision, metadata and success-only publication.
worker.py integrates the pinned baseline and manages the embedding service.
config.py and inputs.py contain settings and input preparation. Inference does
not import the evaluator, which alone has access to reference answers.

    .venv/bin/python -m unittest discover -s tests -t .

Synthetic tests cover label isolation, wrong IDs, symlinks, settings, process
failure/timeouts, output publication and experiment identity. A real single-case
run additionally needs model access, enough memory and a working local server.
