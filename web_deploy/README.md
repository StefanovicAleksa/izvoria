# web_deploy

The hosted, multi-firm, multi-user web version of Izvoria. The full guide is in [docs/web_deploy.md](../docs/web_deploy.md).

Every firm's data (databases, vaults, uploads, backups) lives on the Hetzner VPS. The rented GPU box only computes: it reaches the data through an SSH tunnel it opens, and keeps an ingestion job's files in scratch only while the job runs (see "Where data lives" in the guide).

| Folder | Runs on | What |
|---|---|---|
| `common/` | both | Firm slugs, databases and directories (`tenancy.py`); the gateway-to-box request signature (`signing.py`); upload folders and their cleanup (`uploads.py`); cited sources, located on the box and served on the VPS (`sources.py`). |
| `gateway/` | Hetzner VPS | Firms, users and sessions (SQLite), two-step sign-in (firm, then account), firm administrators, the public request and help pages and their email (`mailer.py`), the signed proxy, uploads and cited sources from the VPS's vault (`data.py`), Vast start/stop, the box's SSH access and idle stop, `admin_cli.py`. Serves the built web app from `gateway/static`. |
| `api/` | GPU box | The HTTP API over the appliance. `engine_host.py` is the only module that imports `app.*`: one shared model kit, one engine per firm. |
| `ingest_worker/` | GPU box | The ingestion wait list (`onyx_control`, in the VPS's Postgres), the supervisor that runs each job in a fresh child process, and the staging of each job's files in from and out to the VPS (`staging.py`). |
| `backup/` | Hetzner VPS | Every firm database and vault, and the gateway database, to a Storage Box with restic; the restore. |
| `uploader/` | the firm's computer | `python -m web_deploy.uploader`: makes each video's 720p copy locally and sends it in resumable parallel pieces. Standard library + ffmpeg. |
| `frontend/` | browser | React + Vite + TypeScript. `npm run build` writes to `gateway/static`. |
| `local/` | dev box | `run_local.py`: the whole thing on one 12 GB dev GPU, on your LAN (see "Local testing on dev hardware" in the guide). |
| `provision/` | both | Box: `provision.sh`, `supervisord.conf`, the tunnel (`box_tunnel.sh`), the llama-server build and launch. VPS: `install_gateway.sh` (Postgres, the data directories, the box's confined SSH account), `add_box_keys.sh`, `izvoria-box-keys`, `Caddyfile`, systemd units, `seed_demo.py`. |
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
