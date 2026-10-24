import asyncio
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import requests

from chimera_agent.config import BASELINE_COMMIT, Settings

# The MCP client starts tool servers with only a minimal default environment; these must reach them.
FORWARDED_ENV = ("CHIMERA_EMBED_SOCKET", "ANONYMIZED_TELEMETRY", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
                 "PYTHONDONTWRITEBYTECODE")


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


def execute(case_dir: Path, settings: Settings, system: str, resource_dir: Path) -> None:
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
    cfg.paths.data_root = str(case_dir / "input")
    cfg.paths.output_dir = str(case_dir / "work/output")
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
        (case_dir / "readiness.json").write_text(json.dumps(identity, indent=2))
        if system == "baseline_scaffold":
            asyncio.run(run_agent(cfg))
        else:
            asyncio.run(run_traced(cfg, case_dir, settings.baseline_dir, system))
    finally:
        service.stop()


def prompts_for(system: str) -> tuple[str, str]:
    # The agent system prompt and the decision guidance; only these differ between the traced systems.
    from chimera_agent_baseline.agent.prompts import build_system_prompt

    from chimera_agent.prompts import AGENT_V2_DECISION_GUIDANCE, AGENT_V2_SYSTEM_PROMPT
    return {"agent_v1": (build_system_prompt(), ""),
            "agent_v2": (AGENT_V2_SYSTEM_PROMPT, AGENT_V2_DECISION_GUIDANCE)}[system]


async def run_traced(cfg, case_dir: Path, baseline_dir: Path, system: str) -> None:
    from chimera_agent_baseline.case_loader import load_cases
    from chimera_agent_baseline.models import load_model
    from langchain_mcp_adapters.client import MultiServerMCPClient

    from chimera_agent.decision import official_output
    from chimera_agent.graph import run_case

    input_dir = case_dir / "input/task1/agent_input"
    (query,) = load_cases(input_dir, task=1, templates_dir=baseline_dir / "templates/prompts")
    client = MultiServerMCPClient({"chimera": {
        "command": sys.executable, "transport": "stdio",
        "args": ["-m", "chimera_agent_baseline.mcp_server", "--data-dir", str(input_dir),
                 "--resource-dir", cfg.paths.resource_dir, "--tool-registry", "task1"],
        "env": {key: os.environ[key] for key in FORWARDED_ENV},
    }})
    tools = await client.get_tools()
    system_prompt, guidance = prompts_for(system)
    trace = await run_case(load_model(cfg), tools, system, system_prompt, guidance, query["case_id"], query["context"],
                           cfg.agent.max_iterations)
    decision, reasoning = official_output(trace.decision, trace.tool_calls)
    output = case_dir / "work/output/task1" / query["case_id"]
    output.mkdir(parents=True)
    (output / "prostate-biopsy-decision.json").write_text(json.dumps(decision))
    (output / "prostate-biopsy-decision-reasoning.json").write_text(json.dumps(reasoning, indent=2))
    (case_dir / "work/trace.json").write_text(trace.model_dump_json(by_alias=True, indent=2))


if __name__ == "__main__":
    directory, system, resources = Path(sys.argv[1]).resolve(), sys.argv[2], Path(sys.argv[3]).resolve()
    settings = Settings(**{
        k: Path(v) if k in ("data_dir", "baseline_dir", "embedding_dir") else v
        for k, v in json.loads((directory / "settings.json").read_text()).items()
    })
    execute(directory, settings, system, resources)
