import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent
BASELINE_COMMIT = "1afa788efc21c0e0144706d75170b0a4cd267367"


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    baseline_dir: Path
    embedding_dir: Path
    base_url: str
    model: str
    temperature: float
    top_p: float
    max_tokens: int
    context_length: int
    request_timeout: int
    max_iterations: int
    wall_seconds: int

    def __post_init__(self):
        url = urlparse(self.base_url)
        if url.scheme != "http" or url.hostname not in ("localhost", "127.0.0.1") or url.path != "/v1":
            raise ValueError("Only a local Ollama http://localhost:<port>/v1 endpoint is allowed")
        if self.model != "gemma4:e2b-it-q8_0":
            raise ValueError("Scaffold requires the baseline model gemma4:e2b-it-q8_0")
        if not 0 <= self.temperature <= 2 or not 0 < self.top_p <= 1:
            raise ValueError("Invalid sampling settings")
        for key in ("max_tokens", "context_length", "request_timeout", "max_iterations", "wall_seconds"):
            value = getattr(self, key)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{key} must be a positive integer")
        if self.request_timeout > self.wall_seconds or self.wall_seconds > 600:
            raise ValueError("Request timeout must fit within wall_seconds <= 600")

    def payload(self) -> dict:
        return {k: str(v) if isinstance(v, Path) else v for k, v in asdict(self).items()}


def load_settings(path: Path) -> Settings:
    data = json.loads(path.read_text())
    for key in ("data_dir", "baseline_dir", "embedding_dir"):
        data[key] = (ROOT / data[key]).resolve()
    return Settings(**data)


def config_id(settings: Settings, identity: dict) -> str:
    values = settings.payload()
    for key in ("data_dir", "baseline_dir", "embedding_dir", "base_url"):
        del values[key]
    encoded = json.dumps({"settings": values, "identity": identity}, sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()
