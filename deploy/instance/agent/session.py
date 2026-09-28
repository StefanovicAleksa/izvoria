import json
import logging
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import psutil

from deploy.instance.agent.config import AgentSettings

logger = logging.getLogger(__name__)

STOP_TIMEOUT_SECONDS = 30.0


class SessionBusy(RuntimeError):
    pass


class UnknownLanguage(ValueError):
    pass


@dataclass(frozen=True)
class RunningApp:
    pid: int
    language: str


def launch_command(settings: AgentSettings) -> List[str]:
    return [str(settings.python), "frontend/main.py", "--env", "production", "--autostart"]


def launch_env(settings: AgentSettings, language: str, base: Dict[str, str]) -> Dict[str, str]:
    """
    Real environment variables beat .env.prod in pydantic-settings, so this is how one checkout serves two languages:
    the database and APPLIANCE_LANGUAGE are chosen here, per launch, and .env.prod holds everything else.
    """
    if language not in settings.databases:
        raise UnknownLanguage(f"'{language}' is not a language this appliance serves.")
    env = dict(base)
    env.update(
        ENV="production",
        APPLIANCE_LANGUAGE=language,
        POSTGRES_DB=settings.databases[language],
        DISPLAY=settings.display,
        QT_QPA_PLATFORM="xcb",
    )
    return env


class AppSession:
    """
    Owns the one desktop-app process on the box. The pid is kept in a state file so a restarted agent still knows
    which process is the appliance instead of launching a second one onto the same GPU.
    """

    def __init__(
        self,
        settings: AgentSettings,
        popen: Callable[..., subprocess.Popen] = subprocess.Popen,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._settings = settings
        self._popen = popen
        self._clock = clock

    def current(self) -> Optional[RunningApp]:
        try:
            data = json.loads(self._settings.state_file.read_text())
            app = RunningApp(pid=int(data["pid"]), language=str(data["language"]))
        except (OSError, ValueError, KeyError):
            return None
        if not _is_alive(app.pid):
            return None
        return app

    def ensure(self, language: str, busy: bool) -> RunningApp:
        running = self.current()
        if running and running.language == language:
            return running
        env = launch_env(self._settings, language, dict(os.environ))
        if running:
            if busy:
                raise SessionBusy(
                    f"The appliance is still working in '{running.language}'. Switch languages once it finishes."
                )
            self.stop()
        self._settings.log_dir.mkdir(parents=True, exist_ok=True)
        with (self._settings.log_dir / f"app-{language}.log").open("ab") as log:
            process = self._popen(
                launch_command(self._settings),
                cwd=str(self._settings.onyx_root),
                env=env,
                start_new_session=True,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        app = RunningApp(pid=process.pid, language=language)
        self._settings.state_file.parent.mkdir(parents=True, exist_ok=True)
        self._settings.state_file.write_text(json.dumps({"pid": app.pid, "language": app.language}))
        logger.info(f"Launched the desktop app (pid {app.pid}) in '{language}'.")
        return app

    def stop(self) -> None:
        """SIGTERM ends the app at once (it installs no handler); process exit hands its VRAM back to the driver."""
        running = self.current()
        if running:
            _terminate(running.pid, self._clock)
        try:
            self._settings.state_file.unlink()
        except FileNotFoundError:
            pass


def _is_alive(pid: int) -> bool:
    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.Error:
        return False


def _terminate(pid: int, clock: Callable[[], float]) -> None:
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = clock() + STOP_TIMEOUT_SECONDS
    while clock() < deadline:
        if not _is_alive(pid):
            return
        time.sleep(0.5)
    logger.warning(f"Desktop app (pid {pid}) ignored SIGTERM; killing it.")
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
