"""
The VPS side of an update (update_fleet.py drives it over SSH): reads and changes the GPU boxes through Vast with the
gateway's own key, which never leaves this machine, and keeps the fleet file (fleet.py) the gateway reads.

Run as root with the gateway's settings loaded, from /opt/onyx_v3:
    set -a; . /etc/onyx-web-gateway.env; set +a
    /opt/onyx-web-gateway/bin/python -m web_deploy.gateway.fleet_cli status
    ... maintenance add|remove <id>     the pool leaves the box alone while it is updated
    ... start <id> | stop <id>          only a box under maintenance can be started here
    ... set-release <commit>            the version every box should hold
    ... record <id> <commit>            the version a box was updated to
    ... health                          the in-use box's /health, through its tunnel

Every command prints one JSON object.
"""

import argparse
import asyncio
import dataclasses
import json
import sys
from typing import List, Optional

import httpx

from web_deploy.gateway import fleet
from web_deploy.gateway.config import GatewaySettings
from web_deploy.gateway.vast_pool import VastPool, build_pool


def _pool(settings: GatewaySettings) -> VastPool:
    return build_pool(settings.vast_api_key, settings.vast_instance_ids, settings.scheduling_grace_seconds,
                      fleet=lambda: fleet.load(settings.fleet_file))


def _known(settings: GatewaySettings, instance: int) -> None:
    if instance not in settings.vast_instance_ids:
        raise SystemExit(json.dumps({"error": f"{instance} is not in VAST_INSTANCE_ID "
                                              f"{list(settings.vast_instance_ids)}"}))


async def status(settings: GatewaySettings, pool: VastPool) -> dict:
    state = fleet.load(settings.fleet_file)
    infos = await pool.describe()
    boxes = []
    for instance in settings.vast_instance_ids:
        info = infos.get(instance)
        boxes.append({
            "id": instance,
            "actual_status": info.actual_status if info else "missing",
            "intended_status": info.intended_status if info else "missing",
            "price_per_hour": info.price_per_hour if info else None,
            "ssh_host": info.public_ip if info else None,
            "ssh_port": info.ports.get(22) if info else None,
            "release": state.boxes.get(instance, ""),
            "current": state.current(instance),
            "maintenance": instance in state.maintenance,
        })
    return {"release": state.release, "boxes": boxes}


def _change(settings: GatewaySettings, **changes) -> dict:
    state = dataclasses.replace(fleet.load(settings.fleet_file), **changes)
    fleet.save(settings.fleet_file, state)
    return state.as_json()


async def run(args: argparse.Namespace, settings: GatewaySettings, pool: VastPool,
              http: Optional[httpx.AsyncClient] = None) -> dict:
    if settings.fleet_file is None:
        raise SystemExit(json.dumps({"error": "No fleet file configured."}))
    state = fleet.load(settings.fleet_file)
    command = args.command
    if command == "status":
        return await status(settings, pool)
    if command == "maintenance":
        _known(settings, args.id)
        updating = set(state.maintenance)
        if args.action == "add":
            updating.add(args.id)
        else:
            updating.discard(args.id)
        return _change(settings, maintenance=frozenset(updating))
    if command == "start":
        _known(settings, args.id)
        if args.id not in state.maintenance:
            # Otherwise the gateway, seeing a second box running, would stop it in the middle of the update
            raise SystemExit(json.dumps({"error": f"Put {args.id} under maintenance before starting it."}))
        reason = await pool.start_one(args.id)
        return {"id": args.id, "started": reason is None, "reason": reason or ""}
    if command == "stop":
        _known(settings, args.id)
        await pool.stop_one(args.id)
        return {"id": args.id, "stopped": True}
    if command == "set-release":
        return _change(settings, release=args.release)
    if command == "record":
        _known(settings, args.id)
        return _change(settings, boxes={**state.boxes, args.id: args.release})
    if command == "health":
        client = http or httpx.AsyncClient()
        try:
            response = await client.get(f"http://127.0.0.1:{settings.box_tunnel_port}/health", timeout=5.0)
            return {"reachable": True, **response.json()}
        except (httpx.HTTPError, ValueError) as exc:
            return {"reachable": False, "error": str(exc)}
        finally:
            if http is None:
                await client.aclose()
    raise SystemExit(json.dumps({"error": f"Unknown command {command}"}))


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="fleet_cli", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("status")
    m = sub.add_parser("maintenance")
    m.add_argument("action", choices=("add", "remove"))
    m.add_argument("id", type=int)
    for name in ("start", "stop"):
        sub.add_parser(name).add_argument("id", type=int)
    sub.add_parser("set-release").add_argument("release")
    r = sub.add_parser("record")
    r.add_argument("id", type=int)
    r.add_argument("release")
    sub.add_parser("health")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = parser().parse_args(argv)
    settings = GatewaySettings.from_env()
    result = asyncio.run(run(args, settings, _pool(settings)))
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
