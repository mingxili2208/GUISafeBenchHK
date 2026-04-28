"""Persistent application state for the GUI console."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
import json
from typing import Any, Dict, Optional

from . import settings


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


@dataclass
class EnvironmentState:
    repo_root: str
    python_exec: str
    carla_host: str
    carla_port: int
    tm_port: int
    carla_session_connected: bool = False
    last_checked_at: Optional[str] = None
    carla_reachable: bool = False
    safebench_import_ok: bool = False
    safebench_module_path: Optional[str] = None
    python_exists: bool = False
    error: Optional[str] = None

    @classmethod
    def default(cls) -> "EnvironmentState":
        return cls(
            repo_root=str(settings.REPO_ROOT),
            python_exec=settings.default_python_exec(),
            carla_host="127.0.0.1",
            carla_port=2000,
            tm_port=8000,
        )

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "EnvironmentState":
        default_state = cls.default()
        merged = asdict(default_state)
        merged.update(payload or {})
        if not settings.is_valid_python_exec(merged.get("python_exec", "")):
            merged["python_exec"] = default_state.python_exec
            merged["carla_session_connected"] = False
            merged["safebench_import_ok"] = False
            merged["safebench_module_path"] = None
            merged["python_exists"] = settings.is_valid_python_exec(default_state.python_exec)
            merged["error"] = "Saved Python interpreter was invalid and has been reset."
        return cls(**merged)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AppStateStore:
    """Small JSON-backed state store for environment selections."""

    def __init__(self, path=None):
        self.path = path or settings.STATE_FILE

    def load(self) -> EnvironmentState:
        if not self.path.exists():
            return EnvironmentState.default()
        with self.path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return EnvironmentState.from_dict(payload)

    def save(self, state: EnvironmentState) -> EnvironmentState:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as handle:
            json.dump(state.to_dict(), handle, ensure_ascii=False, indent=2)
        return state
