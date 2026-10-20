# Canvas: desktop_app_v1
# Path: frontend/izvoria_desktop/service/desktop_app.py

"""
Composition root for the desktop app: environment, logging, gateway, bridge, window.

`apply_runtime_environment()` is separated out and kept free of Qt so it can be unit-tested. What
it does is not cosmetic -- two of its four lines decide whether the app reads the right data at all.
"""

import logging
import os
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# Loggers raised to INFO so the ingest view has something to show. `app` covers every phase,
# feature and the model manager (they all propagate up to it); the others are third-party libraries
# that narrate their own per-file progress and sit outside the `app.*` tree entirely.
_VERBOSE_LOGGERS = ("app", "faster_whisper", "nemo_logger")


def apply_runtime_environment(project_root: Path) -> None:
    """
    Every environment constraint in one place, in the order that matters.

    - `ENV` selects which .env file `app.core.config` loads, and it is read at *import* time, so it
      has to be set before anything imports `app.*`.
    - `PYTORCH_CUDA_ALLOC_CONF` must precede the torch import to prevent fragmentation crashes;
      app/main.py sets the same value at the very top of the file for the same reason.
    - `settings.DATA_DIR` is the relative string "onyx_data", so the vault resolves against the
      process working directory. Launched from anywhere else, the GUI would silently read a
      different vault than the CLI writes to.

    Nothing here concerns media playback: the app has no embedded player. Source files are handed
    to the desktop's own player, which is why Qt's multimedia stack (and its hardware-decode
    crashes on NVIDIA hardware with no usable VAAPI driver) is not in the process at all.
    """
    os.environ.setdefault("ENV", "development")
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.chdir(project_root)


def configure_logging(relay) -> None:
    """Root stays at WARNING so third-party chatter is excluded; the pipeline's tree goes to INFO."""
    logging.basicConfig(level=logging.WARNING)
    relay.attach("app", level=logging.INFO)
    for name in _VERBOSE_LOGGERS[1:]:
        logging.getLogger(name).setLevel(logging.INFO)


def build_gateway(stub: bool):
    """
    The stub is imported unconditionally (it is pure Python); the real gateway is imported only
    when asked for, so `--stub` never pays for the engine's import graph.
    """
    if stub:
        from izvoria_desktop.data.stub_gateway import StubEngineGateway

        sample = _sample_clip()
        if sample:
            logger.info(f"Stub gateway will use {sample} as its sample clip.")
        return StubEngineGateway(clip_path=sample)

    from izvoria_desktop.data.engine_gateway import IzvoriaEngineGateway

    return IzvoriaEngineGateway()


def build_links(stub: bool):
    """
    YouTube links for the Video ingest, or None where yt-dlp isn't installed (an appliance: it never reaches the
    internet). See data/youtube_links.py.
    """
    if stub:
        from izvoria_desktop.data.stub_gateway import StubYouTubeLinks

        return StubYouTubeLinks()

    from izvoria_desktop.data import youtube_links

    if not youtube_links.available():
        return None
    cookies = youtube_links.default_cookies()
    logger.info(f"YouTube links are offered; cookies {'from ' + str(cookies) if cookies.is_file() else 'not set'}.")
    return youtube_links.YouTubeLinks(youtube_links.default_root(), cookies)


def _sample_clip() -> Optional[str]:
    """
    Finds any already-ingested clip so stub mode offers a real file to open rather than a disabled
    button. Best effort: no clip is not an error.
    """
    vault = Path("onyx_data")
    if not vault.exists():
        return None
    for candidate in sorted(vault.rglob("*.mp4"))[:1]:
        return str(candidate.resolve())
    return None


def run(
    argv: List[str],
    stub: bool = False,
    autostart: bool = False,
    mode: Optional[str] = None,
) -> int:
    """`mode` picks light or dark; None falls back to the mode last picked in the app, then ONYX_MODE, then dark."""
    # Imported here, after apply_runtime_environment() has run.
    from PySide6.QtWidgets import QApplication

    from izvoria_desktop.service.engine_bridge import EngineBridge
    from izvoria_desktop.service.log_relay import LogRelay
    from izvoria_desktop.ui import theme
    from izvoria_desktop.ui.main_window import MainWindow

    _assert_single_qt()

    app = QApplication(argv)
    app.setApplicationName("Izvoria")
    app.setOrganizationName("Izvoria")
    from izvoria_desktop.ui.icons import app_icon

    app.setWindowIcon(app_icon())
    from izvoria_desktop.ui import appearance
    from izvoria_desktop.utils import i18n

    # Before any widget exists: widgets read the palette's colours while they build themselves. The in-app
    # switch is the user's latest word, so it beats the environment but not an explicit flag.
    theme.use(mode or appearance.saved_mode() or os.environ.get("ONYX_MODE") or theme.DEFAULT_MODE)
    theme.apply_to(app)
    # Same order for the display language: the choice made in the app, then the environment, then English
    display = appearance.saved_display_language() or os.environ.get("ONYX_DISPLAY_LANGUAGE") or i18n.BASE
    i18n.use(display if display in {language.code for language in i18n.available()} else i18n.BASE)

    relay = LogRelay()
    configure_logging(relay)

    bridge = EngineBridge(build_gateway(stub), links=build_links(stub))
    window = MainWindow(
        bridge=bridge,
        log_relay=relay,
        stub_mode=stub,
    )
    window.show()
    if autostart:
        bridge.boot()

    exit_code = app.exec()

    # Backstop for paths that bypass closeEvent (a session logout, or SIGTERM): without this the
    # process can hang at exit joining the model manager's non-daemon worker thread.
    relay.detach()
    bridge.stop(timeout_ms=15000)
    return exit_code


def _assert_single_qt() -> None:
    """
    Plain `opencv-python` bundles its own Qt plugins and hijacks QT_PLUGIN_PATH, which makes
    PySide6 load the wrong platform plugin and abort. The project pins the headless build
    precisely to avoid this, so a swap should fail loudly here rather than as a cryptic Qt abort.
    """
    try:
        import cv2
    except ImportError:
        return
    bundled = Path(cv2.__file__).parent / "qt"
    if bundled.exists():
        raise RuntimeError(
            "opencv-python bundles Qt plugins that conflict with PySide6. "
            "Install opencv-python-headless instead (this project pins the headless build)."
        )
