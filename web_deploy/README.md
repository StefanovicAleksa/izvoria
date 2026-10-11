# web_deploy

The hosted, multi-firm, multi-user web version of Izvoria. The full guide is in [docs/web_deploy.md](../docs/web_deploy.md).

| Folder | Runs on | What |
|---|---|---|
| `common/` | both | Firm slugs, databases and directories (`tenancy.py`); the gateway-to-box request signature (`signing.py`). Standard library only. |
| `gateway/` | Hetzner VPS | Firms, users and sessions (SQLite), two-step sign-in (firm, then account), firm administrators, the public request and help pages and their email (`mailer.py`), the signed proxy, Vast start/stop and idle stop, `admin_cli.py`. Serves the built web app from `gateway/static`. |
| `api/` | GPU box | The HTTP API over the appliance. `engine_host.py` is the only module that imports `app.*`: one shared model kit, one engine per firm. |
| `ingest_worker/` | GPU box | The ingestion wait list (`onyx_control`), and the supervisor that runs each job in a fresh child process. |
| `frontend/` | browser | React + Vite + TypeScript. `npm run build` writes to `gateway/static`. |
| `local/` | dev box | `run_local.py`: the whole thing on one 12 GB dev GPU, on your LAN (see "Local testing on dev hardware" in the guide). |
| `provision/` | both | Box: `provision.sh`, `supervisord.conf`, the llama-server build and launch, `seed_demo.py`. Gateway: `install_gateway.sh`, `Caddyfile`, systemd unit. |
| `tests/` | dev | `web_deploy/.venv/bin/pytest web_deploy/tests` |

Two rules:

- **Like `deploy/`, this is ops tooling outside the feature-first layout.**
  - It has its own venvs.
  - Never add its dependencies to the root `requirements.txt`: the appliance ships air-gapped.
  - The box's venv is the appliance's own plus `api/requirements.txt`.
- **The modules reused from `deploy/` are imported, not copied:**
  - the Vast client;
  - the vault-path check;
  - the media preparer;
  - upload filename sanitising.

  `deploy/tests` keeps covering them.
