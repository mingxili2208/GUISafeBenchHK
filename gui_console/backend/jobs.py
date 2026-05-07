"""Subprocess job orchestration for long-running GUI tasks."""

from __future__ import annotations

from collections import deque
from datetime import datetime
import json
import os
from pathlib import Path
import shlex
import subprocess
import threading
from typing import Any, Dict, List, Optional
import uuid

from . import settings


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


class JobStore:
    """Persistent job registry backed by JSON sidecars and log files."""

    def __init__(self, root: Optional[Path] = None):
        self.root = root or settings.JOBS_ROOT
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._jobs = {}
        self._load_existing_jobs()

    def _job_path(self, job_id: str) -> Path:
        return self.root / "{job_id}.json".format(job_id=job_id)

    def _job_log_path(self, job_id: str) -> Path:
        return self.root / "{job_id}.log".format(job_id=job_id)

    def _job_control_path(self, job_id: str) -> Path:
        return self.root / "{job_id}.control.json".format(job_id=job_id)

    def _load_existing_jobs(self) -> None:
        for path in sorted(self.root.glob("*.json")):
            if path.name.endswith(".control.json"):
                continue
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if payload.get("status") in {"running", "starting"}:
                pid = payload.get("pid")
                if not pid or not self._pid_exists(pid):
                    payload["status"] = "stale"
                    payload["ended_at"] = payload.get("ended_at") or now_iso()
                    payload["error"] = payload.get("error") or "Backend restarted before job state was refreshed."
                    with path.open("w", encoding="utf-8") as rewritten:
                        json.dump(payload, rewritten, ensure_ascii=False, indent=2)
            self._jobs[payload["id"]] = payload

    @staticmethod
    def _pid_exists(pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True

    def _save_job(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        self._jobs[payload["id"]] = payload
        with self._job_path(payload["id"]).open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        return payload

    def list_jobs(self) -> List[Dict[str, Any]]:
        with self._lock:
            jobs = list(self._jobs.values())
        return sorted(jobs, key=lambda item: item.get("started_at") or "", reverse=True)

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            payload = self._jobs.get(job_id)
        return dict(payload) if payload else None

    def tail_log(self, job_id: str, lines: int = 200) -> List[str]:
        log_path = self._job_log_path(job_id)
        if not log_path.exists():
            return []
        tail = deque(maxlen=max(1, lines))
        with log_path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                tail.append(line.rstrip("\n"))
        return list(tail)

    @staticmethod
    def _summarize_failure_from_lines(lines: List[str]) -> Optional[str]:
        if not lines:
            return None

        patterns = (
            "traceback",
            "exception",
            "error:",
            "runtimeerror",
            "assertionerror",
            "segmentation fault",
            "sigsegv",
            "aborted",
            "killed",
            "critical",
            "failed!",
        )

        for line in reversed(lines):
            stripped = line.strip()
            if not stripped:
                continue
            lower = stripped.lower()
            if any(pattern in lower for pattern in patterns):
                return stripped

        for line in reversed(lines):
            stripped = line.strip()
            if stripped:
                return stripped
        return None

    @staticmethod
    def _wrap_named_command(command: List[str], process_name: Optional[str]) -> List[str]:
        if not process_name:
            return command
        return ["bash", "-lc", 'exec -a "$0" "$@"', process_name, *command]

    def start_job(
        self,
        job_type: str,
        command: List[str],
        cwd: Path,
        metadata: Optional[Dict[str, Any]] = None,
        output_hints: Optional[Dict[str, Any]] = None,
        env: Optional[Dict[str, str]] = None,
        process_name: Optional[str] = None,
        supports_control: bool = False,
    ) -> Dict[str, Any]:
        metadata = metadata or {}
        output_hints = output_hints or {}

        job_id = "job-{stamp}-{suffix}".format(
            stamp=datetime.now().strftime("%Y%m%d-%H%M%S"),
            suffix=uuid.uuid4().hex[:8],
        )
        log_path = self._job_log_path(job_id)
        payload = {
            "id": job_id,
            "type": job_type,
            "status": "starting",
            "command": shlex.join(command),
            "process_name": process_name,
            "cwd": str(cwd),
            "pid": None,
            "started_at": now_iso(),
            "ended_at": None,
            "log_path": str(log_path),
            "return_code": None,
            "metadata": metadata,
            "output_hints": output_hints,
            "supports_control": supports_control,
            "control_requested": None,
            "control_requested_at": None,
            "error": None,
        }
        launch_env = dict(env or os.environ.copy())
        if supports_control:
            control_path = self._job_control_path(job_id)
            if control_path.exists():
                control_path.unlink()
            payload["output_hints"] = {**output_hints, "control_path": str(control_path)}
            launch_env["SAFEBENCH_JOB_CONTROL_PATH"] = str(control_path)
            launch_env["SAFEBENCH_JOB_ID"] = job_id
        with self._lock:
            self._save_job(payload)

        log_handle = log_path.open("a", encoding="utf-8")
        launch_command = self._wrap_named_command(command, process_name)

        # --- DEBUG: enable core dump + heap corruption detection for child process ---
        import resource as _resource
        try:
            _resource.setrlimit(_resource.RLIMIT_CORE, (_resource.RLIM_INFINITY, _resource.RLIM_INFINITY))
        except Exception:
            pass
        # MALLOC_CHECK_=3: glibc detects heap corruption and aborts with SIGABRT (generates core)
        # MALLOC_PERTURB_=0xcd: fills freed memory so use-after-free corrupts predictably near crash site
        launch_env.setdefault("MALLOC_CHECK_", "3")
        launch_env.setdefault("MALLOC_PERTURB_", "205")  # 0xcd
        # --- END DEBUG ---

        try:
            process = subprocess.Popen(
                launch_command,
                cwd=str(cwd),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                env=launch_env,
                start_new_session=True,
            )
        except Exception as exc:
            log_handle.close()
            payload["status"] = "failed"
            payload["ended_at"] = now_iso()
            payload["error"] = "{name}: {message}".format(
                name=type(exc).__name__,
                message=str(exc),
            )
            with self._lock:
                self._save_job(payload)
            return payload

        payload["status"] = "running"
        payload["pid"] = process.pid
        with self._lock:
            self._save_job(payload)

        watcher = threading.Thread(
            target=self._watch_process,
            args=(job_id, process, log_handle),
            daemon=True,
        )
        watcher.start()
        return payload

    def request_control(self, job_id: str, action: str) -> Dict[str, Any]:
        if action not in {"pause", "stop"}:
            raise ValueError("Unsupported control action: {action}".format(action=action))

        with self._lock:
            payload = self._jobs.get(job_id)
            if not payload:
                raise KeyError(job_id)
            if not payload.get("supports_control"):
                raise ValueError("Job does not support control actions.")
            if payload.get("status") not in {"running", "starting"}:
                raise RuntimeError("Only running jobs can receive control actions.")

            requested_at = now_iso()
            payload["control_requested"] = action
            payload["control_requested_at"] = requested_at
            self._save_job(payload)

        control_path = self._job_control_path(job_id)
        with control_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "job_id": job_id,
                    "action": action,
                    "requested_at": requested_at,
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )
        return dict(payload)

    def _watch_process(
        self,
        job_id: str,
        process: subprocess.Popen,
        log_handle,
    ) -> None:
        return_code = process.wait()
        log_handle.close()
        control_path = self._job_control_path(job_id)
        control_payload = None
        if control_path.exists():
            try:
                with control_path.open("r", encoding="utf-8") as handle:
                    control_payload = json.load(handle)
            except Exception:
                control_payload = None
        with self._lock:
            payload = self._jobs[job_id]
            payload["return_code"] = return_code
            payload["ended_at"] = now_iso()
            completed_action = None
            if isinstance(control_payload, dict):
                completed_action = control_payload.get("completed_action")
            if return_code == 0 and completed_action == "pause":
                payload["status"] = "paused"
            elif return_code == 0 and completed_action == "stop":
                payload["status"] = "stopped"
            else:
                payload["status"] = "succeeded" if return_code == 0 else "failed"
            if return_code != 0 and not payload.get("error"):
                error_summary = self._summarize_failure_from_lines(self.tail_log(job_id, 120))
                payload["error"] = error_summary or "Process exited with a non-zero return code: {code}".format(
                    code=return_code
                )
            self._save_job(payload)
