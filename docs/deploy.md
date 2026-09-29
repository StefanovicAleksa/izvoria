# Hosted demo: gateway VPS + on-demand Vast.ai GPU box

Onyx is an on-premise appliance. This is **not** how a client runs it; it's how one tester (and demo viewers) use the real desktop app from a browser without us renting a $200+/month dedicated GPU server.

```
browser ──https──► Gateway VPS (~$5/mo, always on)       Caddy TLS → Starlette (deploy/gateway)
                     /             landing page, login, language, Start / Stop, uploads
                     /api/upload   streams files to the agent
                     /desktop/*    HTTP + WebSocket proxy to noVNC
                     idle monitor  stops the GPU box when nobody's there and nothing runs
                              │ Vast REST API (the key never leaves the VPS)
                              ▼
                   Vast.ai instance, RTX 3090, STOPPED when idle (disk kept, ~$10–15/mo)
                     supervisord: postgres 15 + pgvector · Xvnc :1 (localhost) + XFCE · websockify/noVNC
                                  (token-routed) · agent (deploy/instance/agent, shared secret)
                     agent launches:  venv/bin/python frontend/main.py --env production --autostart
```

`deploy/` is ops tooling. It sits outside the feature-first layout in `app/`, has its own venvs (the appliance's `requirements.txt` never gains Starlette, httpx or psutil), and nothing in `app/` imports it. The appliance side needed one change: `frontend/main.py --autostart`, which boots the engine as soon as the window opens.

## How a session works

1. The tester signs in with the shared password (a signed `SameSite=strict` cookie, which is also the CSRF defence), picks the language the app starts in (**Serbian** or **English**), and clicks **Start appliance**.
2. `ApplianceController` (`deploy/gateway/controller.py`) calls `PUT /api/v0/instances/{id}/ {"state": "running"}`, polls the instance until `actual_status == "running"`, reads the mapped ports for 8090 and 6080, waits for the agent's `/health` (Postgres + X display), then calls `POST /session {language}`.
3. The agent (`deploy/instance/agent/session.py`) launches the desktop app with `APPLIANCE_LANGUAGE=<lang>` and `POSTGRES_DB=onyx` in its environment. Real environment variables override `.env.prod` in pydantic-settings. One database holds every language, so the tester can ingest Serbian and English into the same appliance and ask about either in either language. `OnyxEngine.start()` runs the Alembic migrations itself.
4. **Open appliance** loads `/desktop/vnc.html`. The gateway proxies noVNC's static files and the VNC WebSocket, and appends the websockify token itself. The token routes and authenticates in websockify's `TokenFile` plugin, and it never reaches the browser.
5. Uploads go to `/workspace/uploads/<lang>/<YYYYMMDD-HHMMSS>/` on the box, named for the language picked on the page. Filenames are sanitised, and only the scanner's `ALLOWED_EXTENSIONS` are accepted; a test keeps the two lists in step. The page shows the folder to pick in the **Ingest** view and the language to set there ("Language of these files"). The Ingest view's picker is what the batch actually uses.

**Switching language happens inside the app** (the sidebar's "Answer in"). `POST /session` returns a running app as it is, whatever language it started in, and never relaunches it. A relaunch would lose a running job, and it's no longer needed.

**Busy** means GPU utilisation ≥10% or the app's process tree ≥20% CPU at any point in the last 5 minutes (`activity.py`). CPU counts too, because OCR, document conversion and ffmpeg keep the appliance working with the GPU idle.

**Auto-stop.** `IdleMonitor` stops the box only when all of these hold: no desktop WebSocket is open, no upload is in flight, the box isn't busy, and `IDLE_MINUTES` (default 30) have passed since the last of those. A report left running overnight in a closed tab is therefore never cut off; the idle clock starts when it finishes.

**Stop** calls the agent's `/shutdown`, which SIGTERMs the app. The app installs no handler, so it exits at once and the driver reclaims its VRAM. Then Vast stops the container, and supervisord stops Postgres with SIGINT (a fast, clean shutdown).

## Runbook

### 1. Rent the GPU box (Vast.ai)

- **Template:** image `nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04`, SSH launch mode.
  - Docker options: `-p 8090:8090 -p 6080:6080`.
  - On-start script: `bash /workspace/onyx_v3/deploy/instance/onstart.sh`.
  - The image needs the *devel* variant: `nvcc` builds llama-cpp-python for CUDA.
- **Offer:** 1× RTX 3090, **≥120 GB disk** (51 GB models + venv + databases + uploads), `direct_port_count ≥ 4`, reliability ≥ 0.98, and a CUDA driver that matches the image. Prefer hosts with several 3090s: a stopped instance can only restart on the same machine.
- **On-demand, not interruptible.**

### 2. Put Onyx on it (from the dev box)

```bash
deploy/instance/sync_to_box.sh root@<ssh-host> <ssh-port>          # code + 51 GB of models (rsync, resumable)
ssh -p <ssh-port> root@<ssh-host> bash /workspace/onyx_v3/deploy/instance/provision.sh
ssh -p <ssh-port> root@<ssh-host> bash /workspace/onyx_v3/deploy/instance/onstart.sh   # first start by hand
```

`provision.sh` sets up the box:

- Python 3.11 (deadsnakes), PostgreSQL 15 + pgvector (PGDG), XFCE + TigerVNC + noVNC, `mpv` (the app opens sources in the desktop's player), and fonts.
- The appliance venv, with the CUDA build of llama-cpp-python.
- The agent's own venv.
- `/workspace/secrets.env`, and the `onyx` database (`create_databases.sh`).

It prints `AGENT_SECRET` and `WEBSOCKIFY_TOKEN` for the gateway. Re-run it after `sync_to_box.sh … --code` whenever `requirements.txt` changes. `sync_to_box.sh` lists the models it copies; regenerate that list after adding one (the command is in the script).

### 3. Load the demo set

Open the desktop once through the gateway, or directly via `ssh -L 6080:localhost:6080` and the token URL. Ingest the demo media, one batch per language, with **Language of these files** set for each; every language goes into the same `onyx` database. Then click **Stop**.

**Moving from the two-database demo** (`onyx_sr`, `onyx_sl`): pull, re-run `provision.sh` (it creates `onyx`), restart the agent, and re-ingest the demo material into `onyx` as above. The old databases can then be dropped. Nothing merges them automatically. From here on the box is only ever started from the landing page.

### 4. Set up the gateway VPS

Use any ~$5/mo Debian/Ubuntu VM (Hetzner CX22, DigitalOcean, Vultr). Point an A record for your domain at it, then:

```bash
git clone <this repo> /opt/onyx_v3          # or rsync; only deploy/ is used
sudo bash /opt/onyx_v3/deploy/gateway/install.sh demo.example.com
sudoedit /etc/onyx-gateway.env              # password, AGENT_SECRET, WEBSOCKIFY_TOKEN, VAST_API_KEY, VAST_INSTANCE_ID
sudo systemctl restart onyx-gateway
```

Caddy fetches the Let's Encrypt certificate itself. Send the tester the URL and the password, over separate channels.

### Updating

- **Code:** `sync_to_box.sh … --code` while the box runs, then Stop/Start from the page. On the VPS, `git pull && systemctl restart onyx-gateway`.
- **Logs:** on the box in `/workspace/logs/` (`app.log`, `agent.log`, …); on the VPS via `journalctl -u onyx-gateway`.

## Costs (RTX 3090, September 2026 prices)

| | Rate | Month (150 GPU-hours) |
|---|---|---|
| GPU compute | $0.12–0.25/h, only while started | $18–38 |
| Instance disk (~120 GB), billed while stopped too | ~$0.10–0.15/GB-month | $12–18 |
| Gateway VPS | flat | ~$5 |
| **Total** | | **~$35–60** |

Destroying the instance (not just stopping it) is the only way to stop the disk charge. Re-provisioning afterwards takes about an hour, most of it the model upload.

## Known risks

- **A stopped instance can lose its GPU.** While ours is stopped, another renter can take the card. A start then sits in Vast's `scheduling` state, and the page says **"GPU is currently rented by someone else"** after 60 s. After 15 min it gives up (Vast keeps trying on its own). If this happens often, either copy the instance's disk to another machine with a free 3090 (Vast's *copy* in the console) or rent it as *reserved*.
- **Cold start is ~1–2 min** for the container, plus the engine's own start (migrations, language pack). The page says so.
- **SIGTERM kills the app immediately.** The agent sends it only when the appliance is idle, or on an explicit Stop, which asks for confirmation while busy.
- **The noVNC session is shared.** Two browsers see and drive the same desktop. That's fine for one tester; it isn't multi-user.
- **Not the appliance's security model.** The client's media sits on a rented machine, whereas the product promises that data never leaves the premises. Use the tester's own non-confidential material or our demo set, and say so.

## Tests

Unit tests are in `deploy/tests/`, mocked like the rest of the suite. They run in their own venv; under the appliance venv, `deploy/tests/conftest.py` skips the tree.

```bash
python3.11 -m venv deploy/.venv && deploy/.venv/bin/pip install -r deploy/requirements-dev.txt
deploy/.venv/bin/pytest deploy/tests
```

**Local end-to-end run, no rental.** Run the agent, and the gateway with `FAKE_VAST_HOST=127.0.0.1 GATEWAY_HTTPS_ONLY=0`, on this machine. The gateway then treats localhost as the instance, with ports 8090/6080 unmapped. With a real Xvnc + websockify on :6080, this exercises the whole path; the start and stop calls only flip a flag.
