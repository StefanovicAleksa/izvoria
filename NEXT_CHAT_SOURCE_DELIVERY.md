# Next chat: delivering cited sources to a remote user (plan mode)

Paste the prompt below into a new Claude Code session in plan mode.

---

We host Onyx on a rented GPU box and the user reaches the real desktop app through a browser. The pieces are an always-on gateway VPS (Starlette + Caddy, `deploy/gateway`), a Vast.ai GPU instance running the desktop app under Xvnc/XFCE, an agent on that box (`deploy/instance/agent`), and noVNC proxied by the gateway. Read `docs/deploy.md` and the "Hosted demo" paragraph of `CLAUDE.md` first.

**Problem: a remote user can't actually receive the source behind a citation.** When they click a source in the Chat, Search or Report view, `frontend/onyx_desktop/ui/system_open.py` calls `QDesktopServices.openUrl` on `evidence.clip_path`. That opens the file in `mpv` or a PDF viewer *on the GPU box*, and the user watches it through noVNC:
- **Video has no audio.** noVNC carries none, and the picture arrives at VNC frame rate. A deposition clip is useless that way.
- **Documents can be viewed but not downloaded, saved or printed.**
- Click-to-verify is the product's core promise: every claim traceable to its source, for lawyers and doctors. Remotely, it only half works today.

**What I want planned:** delivering the cited source (video/audio clip, whole recording, PDF/DOCX/TXT/EPUB page, image) to the remote user's **own browser**:
- media plays natively with sound, starting at the cited time;
- documents open at the cited page or line;
- everything has a Download button.

**Direction proposed at the end of the last chat (validate it, don't just adopt it):**
1. **Hosted-mode hook in the desktop app.** When it runs on the hosted box (it already takes `--autostart`; maybe a new flag or env var), "Open source" sends a small request to the local agent instead of opening the file locally: path, `start_sec`/`end_sec`, page or line, `anchor_kind`. The on-premise appliance must keep opening files locally, unchanged.
2. **The agent serves the file**, only from Onyx's own clip and upload directories, never an arbitrary path. It must support HTTP `Range` so long recordings stream and seek.
3. **The gateway page wraps the noVNC view with a "Sources" side panel**, pushed over SSE or WebSocket. It shows an HTML5 player at `#t=<start>`, PDFs via the browser viewer at `#page=N`, and a Download button.
4. **Security:**
   - the existing signed session cookie, plus short-lived signed per-file links;
   - path whitelisting on the agent;
   - the agent's shared secret stays between gateway and agent;
   - nothing the browser sends is trusted as a filesystem path.

**Constraints:**
- `deploy/` is ops tooling: outside the feature-first layout, own venvs, and its dependencies never go into `requirements.txt`. Tests go in `deploy/tests`, run with `deploy/.venv/bin/pytest`. Nothing in `app/` may import `deploy/`.
- `app/` stays server-free. Any desktop-app change lives in `frontend/` with mocked unit tests under `frontend/tests` (`_unit` suffix, no `__init__.py`).
- Citations resolve through `physical_path` / `extracted_file_id` (`app/engine/service/onyx_engine.py` `_resolve_citation_paths`, `frontend/onyx_desktop/data/engine_gateway.py` `_media_kind`). A video chunk with an extracted clip gets the clip. Audio-only sources fall back to the **whole original recording**, so size and bandwidth matter. Document chunks have `start_sec`/`end_sec` = 0.0: check `anchor_kind` and use `app/core/anchors` (`docs/core/anchors.md`).
- Bandwidth goes through a ~$5 VPS and Vast's egress. Consider streaming through the gateway versus a signed direct link to the box's mapped port, and the cost and security of each.
- It's still a single shared noVNC desktop (one tester). Say what would change for multiple users, but don't build that.

**Open question to settle with me early:** is the hosted VPS still a demo for one tester at a time, or is the product moving to hosted multi-user use? If the latter, assess honestly whether noVNC is the right foundation, against a web frontend talking to `OnyxEngine` through an API. I'm not asking to build that now.

**Deliverable:** a plan covering the architecture, the files to change in `deploy/gateway`, `deploy/instance/agent` and `frontend/`, the security model, how each source type is presented, tests (unit + a local end-to-end run using `FAKE_VAST_HOST` as described in `docs/deploy.md`), and what to verify on a real Vast box.
