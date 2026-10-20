"""
What the gateway knows about its GPU boxes beyond Vast's view: which version of the app each one holds, which version
is current, and which boxes are being updated right now.

Written by `fleet_cli.py`, which `update_fleet.py` drives over SSH while it deploys, and read by the pool
(vast_pool.py) at every wake:
- a box being updated ("maintenance") is never started, stopped or used: the update started it and stops it again;
- a box holding an older version than the current one is woken only when no up-to-date box can start, so a stale
  copy of the app answers only when the alternative is no answer at all.

One small JSON file (FLEET_FILE, beside the gateway database). A missing or unreadable file means "no versions known",
and every box counts as current: the gateway behaves as before versions existed.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, FrozenSet, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FleetState:
    release: str = ""                                          # the current version; "" when none was deployed
    boxes: Dict[int, str] = field(default_factory=dict)        # instance id -> the version it holds
    maintenance: FrozenSet[int] = frozenset()                  # being updated: the pool leaves them alone

    def current(self, instance: int) -> bool:
        """Whether the box holds the current version (every box does while no version is known)."""
        return not self.release or self.boxes.get(instance) == self.release

    def as_json(self) -> dict:
        return {"release": self.release, "boxes": {str(k): v for k, v in sorted(self.boxes.items())},
                "maintenance": sorted(self.maintenance)}


def load(path: Optional[Path]) -> FleetState:
    if path is None:
        return FleetState()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return FleetState(
            release=str(raw.get("release") or ""),
            boxes={int(k): str(v) for k, v in (raw.get("boxes") or {}).items()},
            maintenance=frozenset(int(i) for i in raw.get("maintenance") or ()),
        )
    except FileNotFoundError:
        return FleetState()
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        logger.error(f"Could not read the fleet file {path}; treating every box as current: {exc}")
        return FleetState()


def save(path: Path, state: FleetState) -> None:
    """Atomically, world-readable: the gateway runs as another user than the root that writes it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".tmp")
    partial.write_text(json.dumps(state.as_json(), indent=2) + "\n", encoding="utf-8")
    os.chmod(partial, 0o644)
    partial.replace(path)
