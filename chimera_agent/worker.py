import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import requests

from chimera_agent.config import BASELINE_COMMIT, Settings


def git_output(path: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


def check_assets(settings: Settings) -> dict:
    baseline = settings.baseline_dir
    dirty = git_output(baseline, "-c", "filter.lfs.process=", "-c", "filter.lfs.required=false",
                       "status", "--porcelain", "--", ".", ":(exclude)**/__pycache__/**", ":(exclude)model/embedding_model/**")
    if dirty:
        raise ValueError("Baseline source checkout has local changes")
    if git_output(baseline, "rev-parse", "HEAD") != BASELINE_COMMIT:
        raise ValueError("Baseline checkout does not match the pinned commit")
    for path in (baseline / "templates/prompts/agent_prompt.j2",
                 baseline / "configs/config.yaml", baseline / "resources/guidelines_db",
                 settings.embedding_dir / "config.json"):
        if not path.exists():
            raise FileNotFoundError(f"Required asset missing: {path}")
    api = settings.base_url.removesuffix("/v1")
    show = requests.post(api + "/api/show", json={"model": settings.model}, timeout=10)
    show.raise_for_status()
    detail = show.json()
    if "tools" not in detail.get("capabilities", []) or detail.get("remote_host"):
        raise ValueError("A local tool-capable model is required")
    tags = requests.get(api + "/api/tags", timeout=10)
    tags.raise_for_status()
    digest = next(m["digest"] for m in tags.json()["models"] if m["name"] == settings.model)
    # No patient information is sent during model/context readiness.
    warm = requests.post(api + "/api/generate", json={
        "model": settings.model, "prompt": "Reply OK.", "stream": False,
        "options": {"num_predict": 1},
    }, timeout=settings.request_timeout)
    warm.raise_for_status()
    loaded = requests.get(api + "/api/ps", timeout=10)
    loaded.raise_for_status()
    model = next(m for m in loaded.json()["models"] if m["name"] == settings.model)
    if model.get("context_length") != settings.context_length:
        raise ValueError("Ollama context mismatch; start with OLLAMA_CONTEXT_LENGTH=16384")
    return {"model_digest": digest, "context_length": model["context_length"]}


def execute(run_dir: Path, settings: Settings) -> None:
    identity = check_assets(settings)
    embedding_config = json.loads((settings.embedding_dir / "config.json").read_text())
    identity["embedding_revision"] = embedding_config.get("_commit_hash", "unknown")
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["ANONYMIZED_TELEMETRY"] = "False"
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["CHIMERA_EMBED_SOCKET"] = str(Path(tempfile.gettempdir()) / f"chm-{os.getpid()}.sock")
    os.environ["CHIMERA_LOCAL_API_KEY"] = "local-placeholder"
    from omegaconf import OmegaConf
    import chimera_agent_baseline
    from chimera_agent_baseline.rag import GuidelinesSearch, start_embedding_service
    from chimera_agent_baseline.run import run_agent

    installed = Path(chimera_agent_baseline.__file__).resolve()
    if not installed.is_relative_to(settings.baseline_dir):
        raise ValueError("Installed baseline does not resolve to the pinned checkout")
    os.chdir(settings.baseline_dir)
    cfg = OmegaConf.load(settings.baseline_dir / "configs/config.yaml")
    cfg.paths.data_root = str(run_dir / "input")
    cfg.paths.output_dir = str(run_dir / "work/output")
    resource_dir = run_dir / "resources"
    # Chroma writes runtime SQLite metadata even for retrieval-only queries.
    shutil.copytree(settings.baseline_dir / "resources", resource_dir)
    cfg.paths.resource_dir = str(resource_dir)
    cfg.paths.embedding_model_dir = str(settings.embedding_dir)
    cfg.agent.tasks = [1]
    cfg.agent.predictor.enabled = False
    cfg.agent.max_iterations = settings.max_iterations
    cfg.agent.step_timeout = settings.wall_seconds
    cfg.model.provider = "openai"
    cfg.model.model_id = settings.model
    cfg.model.base_url = settings.base_url
    cfg.model.api_key_env = "CHIMERA_LOCAL_API_KEY"
    cfg.model.request_timeout = settings.request_timeout
    cfg.generation.temperature = settings.temperature
    cfg.generation.top_p = settings.top_p
    cfg.generation.max_new_tokens = settings.max_tokens
    service = start_embedding_service(settings.embedding_dir)
    if service is None:
        raise RuntimeError("Embedding service did not start")
    try:
        hits = GuidelinesSearch(resource_dir).query("When is prostate biopsy recommended?")
        if not hits or not all(h.get("text") and isinstance(h.get("score"), (int, float)) for h in hits):
            raise RuntimeError("Guideline retrieval returned no real passages")
        identity["retrieval_hits"] = len(hits)
        (run_dir / "readiness.json").write_text(json.dumps(identity, indent=2))
        awaitable = run_agent(cfg)
        asyncio.run(awaitable)
    finally:
        service.stop()


if __name__ == "__main__":
    directory = Path(sys.argv[1]).resolve()
    settings = Settings(**{
        k: Path(v) if k in ("data_dir", "baseline_dir", "embedding_dir") else v
        for k, v in json.loads((directory / "settings.json").read_text()).items()
    })
    execute(directory, settings)
