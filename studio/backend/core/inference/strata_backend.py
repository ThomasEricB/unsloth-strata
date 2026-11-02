# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Runs models from a local Strata installation (https://github.com/Niko1221/Strata) as
``strata:<model_name>``: Studio starts a private ``strata serve`` on authenticated loopback and
chat goes through the managed provider proxy, so Studio's tools (browser use, MCP, search) run
on top. Strata keeps its own tuning: each model is one of Strata's run configs, served as-is.
Does not import the GPU stack."""

from __future__ import annotations

import json
import logging
import os
import re
import secrets
import shutil
import socket
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Optional

import httpx

from core.inference.npu_backend import ManagedUpstream
from utils.native_path_leases import child_env_without_native_path_secret
from utils.process_lifetime import (
    adopt_pid,
    child_popen_kwargs,
    forget_pid,
    is_process_shutting_down,
    spawn_on_lifetime_thread,
    terminate_pid,
)

logger = logging.getLogger(__name__)

MODEL_PREFIX = "strata:"
PROVIDER_TYPE = "strata"

# A large MoE pack loads tens of GB of experts; Strata's own engine wait is 900 s by default.
_LOAD_TIMEOUT_S = float(os.environ.get("UNSLOTH_STRATA_LOAD_TIMEOUT", "1800"))
# SIGTERM lets `strata serve` drain and send QUIT to the engine before the tree is killed.
_STOP_GRACE_S = 30.0
# The engine keeps most of its resident budget (the experts) in RAM. Under memory pressure
# systemd-oomd kills whatever holds the most swap, which may be the desktop session.
_RAM_SHARE_OF_BUDGET = 0.8
_GIB = 1024**3

_TRANSPORT_ERRORS = (
    httpx.ConnectError,
    httpx.ReadError,
    httpx.RemoteProtocolError,
    httpx.WriteError,
    httpx.TimeoutException,
)
_PORT_TAKEN = re.compile(r"address already in use|errno 98\b", re.IGNORECASE)


class StrataError(RuntimeError):
    """A Strata operation failed; the message is safe to show the user."""


class StrataLoadCancelled(StrataError):
    """A load was stopped by cancel_load."""


@dataclass(frozen = True)
class StrataModel:
    id: str
    config_path: str
    max_context_length: Optional[int]
    vision: bool
    resident_budget_gib: Optional[float]
    pack_dir: Optional[str]
    # The run config's own port, where Strata's tools look for a running server.
    port: Optional[int] = None
    # Strata serves tool calls and reasoning for every model family it supports.
    tools: bool = True
    reasoning: bool = True

    @property
    def model_path(self) -> str:
        return MODEL_PREFIX + self.id

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "model_path": self.model_path,
            "config_path": self.config_path,
            "supports_vision": self.vision,
            "supports_reasoning": self.reasoning,
            "supports_tools": self.tools,
            "max_context_length": self.max_context_length,
            "resident_budget_gib": self.resident_budget_gib,
        }


@dataclass(frozen = True)
class StrataResident:
    """One consistent read of the loaded model, taken without the lock long operations hold."""

    model: StrataModel
    context_length: Optional[int]
    requested_context_length: Optional[int]
    base_url: str
    api_key: str


def is_strata_model_path(model_path: Optional[str]) -> bool:
    return isinstance(model_path, str) and model_path.startswith(MODEL_PREFIX)


def model_id_from_path(model_path: str) -> str:
    model_id = model_path[len(MODEL_PREFIX) :].strip()
    if not model_id or "/" in model_id or "\\" in model_id:
        raise StrataError(f"Invalid Strata model id: {model_path!r}")
    return model_id


def _data_home() -> Path:
    return Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")


def install_dirs() -> list[Path]:
    """Where Strata keeps its run configs (strata-<model>.json): the installs its app remembers,
    its default install, and any extra ones in UNSLOTH_STRATA_INSTALL_DIRS."""
    dirs: list[Path] = []
    extra = os.environ.get("UNSLOTH_STRATA_INSTALL_DIRS", "")
    dirs += [Path(p).expanduser() for p in extra.split(os.pathsep) if p.strip()]
    settings = _data_home() / "strata" / "app-settings.json"
    try:
        remembered = json.loads(settings.read_text(encoding = "utf-8")).get("installs") or []
        dirs += [Path(p).expanduser() for p in remembered if isinstance(p, str) and p.strip()]
    except (OSError, ValueError, AttributeError):
        pass
    dirs.append(_data_home() / "strata" / "install")
    unique: list[Path] = []
    for d in dirs:
        if d not in unique:
            unique.append(d)
    return unique


def strata_binary() -> Optional[Path]:
    configured = os.environ.get("UNSLOTH_STRATA_BIN", "").strip()
    found = configured or shutil.which("strata")
    return Path(found) if found and Path(found).is_file() else None


def _flag(args: list, name: str) -> Optional[str]:
    try:
        index = args.index(name)
    except ValueError:
        return None
    return str(args[index + 1]) if index + 1 < len(args) else None


def _positive_int(value: Any) -> Optional[int]:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _positive_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def read_run_config(path: Path) -> Optional[StrataModel]:
    """A run config as a model, or None when it is not one Strata can serve."""
    try:
        config = json.loads(path.read_text(encoding = "utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(config, dict):
        return None
    name = str(config.get("model_name") or "").strip()
    args = config.get("args")
    if not name or "/" in name or "\\" in name or not isinstance(args, list):
        return None
    return StrataModel(
        id = name,
        config_path = str(path),
        max_context_length = _positive_int(_flag(args, "--max-context")),
        vision = bool(config.get("vision")) or "--vision" in args,
        resident_budget_gib = _positive_float(_flag(args, "--resident-budget-gib")),
        pack_dir = _flag(args, "--pack"),
        port = _positive_int(config.get("port")),
    )


def _mem_available_bytes() -> Optional[int]:
    try:
        with open("/proc/meminfo", encoding = "utf-8") as meminfo:
            for line in meminfo:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def running_engines(exclude: frozenset[int] = frozenset()) -> list[int]:
    """PIDs of Strata engines running on this machine (the Strata app's, a terminal's), other
    than ``exclude``. An engine holds the GPU and tens of GB of RAM, so a second one fails or
    pushes the machine into swap."""
    found = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) in exclude:
            continue
        try:
            exe = Path(os.readlink(entry / "exe"))
        except OSError:
            continue
        # <...>/engine/<version>-<platform>-<backend>-<variant>/strata
        if exe.name == "strata" and exe.parent.parent.name == "engine":
            found.append(int(entry.name))
    return sorted(found)


def _kill_with_the_parent() -> None:
    """SIGKILL `strata serve` if Studio dies; its own lifeline then ends the engine."""
    import ctypes
    import signal

    ctypes.CDLL("libc.so.6", use_errno = True).prctl(1, signal.SIGKILL)


class StrataServer:
    """One private `strata serve --profile <run config>` on a random loopback port and key."""

    def __init__(self, binary: Path, config_path: str, preferred_port: Optional[int] = None) -> None:
        self.binary = binary
        self.config_path = config_path
        self.preferred_port = preferred_port
        self.port: Optional[int] = None
        self.api_key: Optional[str] = None
        self.health: dict[str, Any] = {}
        self._process: Optional[subprocess.Popen] = None
        self._tail: deque[str] = deque(maxlen = 200)
        self._events: deque[dict] = deque(maxlen = 50)
        self._threads: list[threading.Thread] = []
        self._stop_requested = threading.Event()
        self._client = httpx.Client(timeout = 10.0, trust_env = False)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def pid(self) -> Optional[int]:
        return self._process.pid if self._process is not None else None

    def is_alive(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def log_tail(self, lines: int = 25) -> str:
        return "\n".join(list(self._tail)[-lines:])

    @staticmethod
    def _free_port(wanted: int = 0) -> Optional[int]:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", wanted))
                return int(s.getsockname()[1])
        except OSError:
            return None

    def _pick_port(self, attempt: int) -> int:
        """The run config's port when it is free: the Strata app and `strata status` look for a
        running server there (and on no random one), so they see this one and do not start a
        second engine beside it. Any free port otherwise, and on a retry."""
        if attempt == 0 and self.preferred_port:
            port = self._free_port(self.preferred_port)
            if port is not None:
                return port
            logger.info("Port %s is taken; serving Strata on a random port.", self.preferred_port)
        port = self._free_port()
        if port is None:
            raise StrataError("No free local port for the Strata server.")
        return port

    def start(self, cancelled: threading.Event, timeout: float = _LOAD_TIMEOUT_S) -> None:
        for attempt in range(3):
            if self._start_once(cancelled, timeout, attempt):
                return
            if cancelled.is_set() or self._stop_requested.is_set():
                raise StrataLoadCancelled("Loading the Strata model was cancelled.")
            if attempt < 2 and _PORT_TAKEN.search(self.log_tail()):
                logger.warning("Strata lost its port to another process; retrying on a new one.")
                continue
            failed = next((e for e in reversed(self._events) if e.get("event") == "failed"), None)
            reason = f" (exit code {failed.get('exit_code')})" if failed else ""
            raise StrataError(f"Strata did not start{reason}. Last output:\n{self.log_tail()}")

    def _start_once(self, cancelled: threading.Event, timeout: float, attempt: int = 0) -> bool:
        if is_process_shutting_down():
            raise StrataError("Unsloth is shutting down; not starting Strata.")
        port, api_key = self._pick_port(attempt), secrets.token_urlsafe(32)
        env = child_env_without_native_path_secret()
        # The key travels in the environment, never argv, which any local user can read.
        env["STRATA_API_KEY"] = api_key
        cmd = [
            str(self.binary),
            "serve",
            "--profile",
            self.config_path,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--output",
            "jsonl",
            "--shutdown-timeout",
            str(int(_STOP_GRACE_S) - 5),
        ]
        logger.info("Starting Strata: %s", " ".join(cmd))
        self._tail.clear()
        self._events.clear()
        self.health = {}

        def _spawn() -> subprocess.Popen:
            return subprocess.Popen(
                cmd,
                stdout = subprocess.PIPE,
                stderr = subprocess.PIPE,
                stdin = subprocess.DEVNULL,
                text = True,
                encoding = "utf-8",
                errors = "replace",
                env = env,
                **child_popen_kwargs(_kill_with_the_parent),
            )

        proc = spawn_on_lifetime_thread(_spawn)
        adopt_pid(proc.pid)
        self._process, self.port, self.api_key = proc, port, api_key
        if is_process_shutting_down():
            self.stop()
            raise StrataError("Unsloth is shutting down; not starting Strata.")
        self._threads = [
            threading.Thread(target = self._read_events, args = (proc,), daemon = True, name = "strata-events"),
            threading.Thread(target = self._read_log, args = (proc,), daemon = True, name = "strata-log"),
        ]
        for thread in self._threads:
            thread.start()
        if self._wait_loaded(cancelled, timeout):
            return True
        self.stop()
        return False

    def _read_events(self, proc: subprocess.Popen) -> None:
        """`--output jsonl`: lifecycle events on stdout (starting, ready, failed, stopped)."""
        try:
            assert proc.stdout is not None
            for raw in proc.stdout:
                line = raw.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except ValueError:
                    self._tail.append(line)
                    continue
                if isinstance(event, dict):
                    self._events.append(event)
                    logger.info("[strata] %s", event.get("event"))
        except Exception:  # noqa: BLE001 -- the pipe closes at teardown
            pass

    def _read_log(self, proc: subprocess.Popen) -> None:
        try:
            assert proc.stderr is not None
            for raw in proc.stderr:
                line = raw.rstrip()
                if line:
                    self._tail.append(line)
                    logger.debug("[strata] %s", line)
        except Exception:  # noqa: BLE001
            pass

    def _wait_loaded(self, cancelled: threading.Event, timeout: float) -> bool:
        """Until /health says the model is loaded; the server answers before the engine is ready."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not self.is_alive() or self._stop_requested.is_set() or cancelled.is_set():
                return False
            if any(e.get("event") == "failed" for e in self._events):
                return False
            try:
                response = self._client.get(f"{self.base_url}/health", timeout = 2.0)
                if response.status_code == 200:
                    body = response.json()
                    if isinstance(body, dict) and body.get("loaded"):
                        self.health = body
                        return True
            except (*_TRANSPORT_ERRORS, ValueError):
                pass
            time.sleep(0.5)
        self._tail.append(f"Strata did not load the model within {int(timeout)} s.")
        return False

    def stop(self) -> None:
        """SIGTERM for a graceful drain, then the whole tree. Idempotent."""
        self._stop_requested.set()
        proc = self._process
        if proc is None:
            return
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout = _STOP_GRACE_S)
                except subprocess.TimeoutExpired:
                    logger.warning("Strata did not stop within %s s; killing it.", _STOP_GRACE_S)
            terminate_pid(proc.pid, timeout = 10.0, owner_verified = True)
            proc.wait(timeout = 10)
        except Exception as exc:  # noqa: BLE001 -- best-effort teardown
            logger.warning("Error stopping Strata: %s", exc)
        finally:
            if proc.poll() is not None:
                forget_pid(proc.pid)
            self._process, self.port, self.api_key = None, None, None
            for thread in self._threads:
                thread.join(timeout = 2)
            self._threads = []

    def close(self) -> None:
        self.stop()
        try:
            self._client.close()
        except Exception:  # noqa: BLE001
            pass


@dataclass
class _Loaded:
    model: StrataModel
    context_length: Optional[int]
    requested_context_length: Optional[int]


class StrataBackend:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        # Makes cancel_load and a load's commit exclusive, so a cancel either wins or finds nothing.
        self._commit_lock = threading.Lock()
        self._server: Optional[StrataServer] = None
        self._loaded: Optional[_Loaded] = None
        self._loading: Optional[str] = None
        self._load_cancelled = threading.Event()
        self._error: Optional[str] = None

    # -- state ---------------------------------------------------------------------------------

    def resident(self) -> Optional[StrataResident]:
        loaded, server = self._loaded, self._server
        if loaded is None or server is None or not server.is_alive():
            return None
        port, api_key = server.port, server.api_key
        if port is None:
            return None
        return StrataResident(
            model = loaded.model,
            context_length = loaded.context_length,
            requested_context_length = loaded.requested_context_length,
            base_url = f"http://127.0.0.1:{port}",
            api_key = api_key or "",
        )

    @property
    def is_loaded(self) -> bool:
        return self.resident() is not None

    @property
    def loaded_model(self) -> Optional[StrataModel]:
        resident = self.resident()
        return resident.model if resident is not None else None

    @property
    def loading_model(self) -> Optional[str]:
        return self._loading

    def status(self) -> dict[str, Any]:
        binary = strata_binary()
        resident = self.resident()
        server = self._server
        return {
            "installed": binary is not None,
            "binary": str(binary) if binary else None,
            "install_dirs": [str(d) for d in install_dirs()],
            "loaded_model": resident.model.model_path if resident else None,
            "context_length": resident.context_length if resident else None,
            "loading_model": self._loading,
            "server_pid": server.pid if server is not None and server.is_alive() else None,
            "other_engines": running_engines(self._own_pids()),
            "error": self._error,
        }

    def _own_pids(self) -> frozenset[int]:
        server = self._server
        if server is None or server.pid is None:
            return frozenset()
        own = {server.pid}
        for entry in Path("/proc").iterdir():
            if entry.name.isdigit():
                try:
                    stat = (entry / "stat").read_text().rsplit(")", 1)[1].split()
                    if int(stat[1]) in own:
                        own.add(int(entry.name))
                except (OSError, IndexError, ValueError):
                    continue
        return frozenset(own)

    # -- catalog -------------------------------------------------------------------------------

    def catalog(self) -> list[StrataModel]:
        models: dict[str, StrataModel] = {}
        for directory in install_dirs():
            for path in sorted(directory.glob("strata-*.json")):
                model = read_run_config(path)
                if model is not None and model.id not in models:
                    models[model.id] = model
        return sorted(models.values(), key = lambda m: m.id)

    def loadable_model(self, model_id: str) -> StrataModel:
        """The catalog entry ``load`` would accept, checked before a caller unloads anything."""
        if strata_binary() is None:
            raise StrataError(
                "Strata is not installed: no `strata` command was found. Install Strata, or set "
                "UNSLOTH_STRATA_BIN to its `strata` launcher."
            )
        for model in self.catalog():
            if model.id == model_id:
                if model.pack_dir and not Path(model.pack_dir).is_dir():
                    raise StrataError(
                        f"The files of {model_id} are missing ({model.pack_dir}). Repair the model "
                        "in Strata first."
                    )
                return model
        raise StrataError(
            f"{model_id} is not one of Strata's installed models. Set it up in Strata first."
        )

    def _refuse_other_engines(self) -> None:
        """Before anything is unloaded: a second engine would not fit beside the first."""
        others = running_engines(self._own_pids())
        if others:
            raise StrataError(
                "Another Strata engine is already running (pid "
                + ", ".join(map(str, others))
                + "), most likely in the Strata app. Unload the model there first: two engines "
                "do not fit in the GPU and RAM together."
            )

    def _refuse_short_memory(self, model: StrataModel) -> None:
        """After the previous model is gone, so its own RAM counts as available."""
        if os.environ.get("UNSLOTH_STRATA_SKIP_MEMORY_CHECK", "").strip() in ("1", "true", "yes"):
            return
        available = _mem_available_bytes()
        if model.resident_budget_gib and available is not None:
            needed = model.resident_budget_gib * _RAM_SHARE_OF_BUDGET * _GIB
            if available < needed:
                raise StrataError(
                    f"{model.id} keeps about {needed / _GIB:.0f} GB of its experts in RAM, but only "
                    f"{available / _GIB:.0f} GB is available. Close other programs first, or lower "
                    "the model's RAM budget in Strata. (UNSLOTH_STRATA_SKIP_MEMORY_CHECK=1 skips "
                    "this check.)"
                )

    # -- lifecycle -----------------------------------------------------------------------------

    def load(self, model_id: str, context_length: Optional[int] = None) -> StrataModel:
        """Serve one model with Strata's own settings; ``context_length`` is recorded, not applied:
        Strata sizes the context from the run config."""
        with self._lock:
            with self._commit_lock:
                self._load_cancelled.clear()
                self._loading = model_id
            loaded = False
            try:
                model = self.loadable_model(model_id)
                binary = strata_binary()
                assert binary is not None
                self._refuse_other_engines()
                self._stop_server()
                self._raise_if_load_cancelled(model_id)
                self._refuse_short_memory(model)
                server = StrataServer(binary, model.config_path, model.port)
                self._server = server
                try:
                    server.start(self._load_cancelled)
                except StrataLoadCancelled:
                    raise
                except StrataError:
                    self._raise_if_load_cancelled(model_id)
                    raise
                ctx = _positive_int(server.health.get("max_context")) or model.max_context_length
                with self._commit_lock:
                    self._raise_if_load_cancelled(model_id)
                    self._loaded = _Loaded(
                        model = replace(model, vision = bool(server.health.get("images", model.vision))),
                        context_length = ctx,
                        requested_context_length = (
                            context_length if context_length and context_length > 0 else None
                        ),
                    )
                    self._loading = None
                    loaded = True
                self._error = None
                return self._loaded.model
            except StrataError as exc:
                self._error = str(exc)
                raise
            finally:
                self._loading = None
                if not loaded:
                    self._stop_server()

    def _raise_if_load_cancelled(self, model_id: str) -> None:
        if self._load_cancelled.is_set():
            raise StrataLoadCancelled(f"Loading {model_id} was cancelled.")

    def cancel_load(self, model_id: Optional[str] = None) -> bool:
        """Stop an in-flight load. Does not take the lock the load holds."""
        with self._commit_lock:
            loading = self._loading
            if loading is None or (model_id is not None and loading != model_id):
                return False
            self._load_cancelled.set()
        server = self._server
        if server is not None:
            server.stop()
        return True

    def _stop_server(self) -> None:
        server, self._server = self._server, None
        self._loaded = None
        if server is not None:
            server.close()

    def unload(self) -> Optional[str]:
        """Stop the resident model. Returns its model path, or None if nothing was loaded."""
        with self._lock:
            loaded = self._loaded
            self._stop_server()
            return loaded.model.model_path if loaded is not None else None

    def shutdown(self) -> None:
        self.cancel_load()
        with self._lock:
            self._stop_server()

    def upstream(self) -> ManagedUpstream:
        # Lock-free: a chat must not wait behind a load holding the lock.
        resident = self.resident()
        if resident is None:
            raise StrataError("No Strata model is loaded.")
        model = resident.model
        return ManagedUpstream(
            provider_type = PROVIDER_TYPE,
            base_url = f"{resident.base_url}/v1",
            api_key = resident.api_key,
            model = model.id,
            public_model = model.model_path,
            supports_vision = model.vision,
            supports_tools = model.tools,
            supports_reasoning = model.reasoning,
            context_length = resident.context_length,
        )


_backend: Optional[StrataBackend] = None
_backend_lock = threading.Lock()


def get_strata_backend() -> StrataBackend:
    global _backend
    with _backend_lock:
        if _backend is None:
            _backend = StrataBackend()
        return _backend


def peek_strata_backend() -> Optional[StrataBackend]:
    """The backend if anything has touched it, without creating one (for shutdown)."""
    return _backend
