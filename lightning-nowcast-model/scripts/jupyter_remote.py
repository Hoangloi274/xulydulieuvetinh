# /// script
# requires-python = ">=3.11"
# dependencies = ["requests", "websocket-client"]
# ///
"""Work on the training server through its JupyterLab: run commands, upload and download files.

The server is a container behind the lab's Jupyter proxy (no SSH). This uses the same HTTPS API as the browser:
the contents API for uploads, /files/ for downloads and a terminal websocket for commands. Run it with `uv run`, which
installs its dependencies (see the header). Configuration comes from the environment, or else from
~/.config/jupyter_remote/env (KEY=VALUE lines; must be readable by you only, chmod 600; keep it out of any project folder):

    JUPYTER_URL       the server base, e.g. https://ws.gvlab.org/fablab/thanhan   (the part before /lab)
    JUPYTER_TOKEN     an API token (JupyterHub: hub/token page); it gives full shell access: revoke it when done
    JUPYTER_PASSWORD  or the server's login password (used when there is no token): the tool logs in like the
                      browser and sends the session cookie and its _xsrf value

    uv run scripts/jupyter_remote.py run "nvidia-smi; df -h /root"            # output streams back; exit code is returned
    uv run scripts/jupyter_remote.py run --timeout 3600 "cd procedure/nowcast-fusion-model && bash scripts/server_train.sh check"
    uv run scripts/jupyter_remote.py upload data/upload/4_packed_r300_ltg.zip procedure/
    uv run scripts/jupyter_remote.py download procedure/nowcast-fusion-model/runs/train/log.jsonl runs/train/
    uv run scripts/jupyter_remote.py ls procedure/nowcast-fusion-model/runs

Remote paths are relative to the Jupyter root (the folder the file browser shows as "/"); commands start there.
Uploads go in parts (--part-mb) of small chunks (--chunk-mb, as the browser's upload does), several parts at a
time; parts already complete on the server are skipped, so an interrupted upload resumes. The parts are then joined
on the server and the file's sha256 is checked against the local one.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import posixpath
import re
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote

import requests
import websocket

ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07]*\x07|\x1b[=>]")


CONFIG = Path.home() / ".config" / "jupyter_remote" / "env"


def settings() -> dict[str, str]:
    env = {k: os.environ[k] for k in ("JUPYTER_URL", "JUPYTER_TOKEN", "JUPYTER_PASSWORD") if os.environ.get(k)}
    if CONFIG.exists():
        if CONFIG.stat().st_mode & 0o077:
            raise SystemExit(f"{CONFIG} is readable by others: chmod 600 {CONFIG}")
        for line in CONFIG.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                env.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    return env


class Jupyter:
    def __init__(self) -> None:
        cfg = settings()
        url, token, password = cfg.get("JUPYTER_URL", ""), cfg.get("JUPYTER_TOKEN", ""), cfg.get("JUPYTER_PASSWORD", "")
        if not (url and (token or password)):
            raise SystemExit(f"set JUPYTER_URL and JUPYTER_TOKEN or JUPYTER_PASSWORD, in the environment or {CONFIG}")
        self.base = re.sub(r"/(lab|tree|hub|login)(/.*)?$", "", url.rstrip("/"))
        self.token = token
        self.http = requests.Session()
        if token:
            self.http.headers["Authorization"] = f"token {token}"
        else:
            self._login(password)

    def _login(self, password: str) -> None:
        """Jupyter's password form: GET /login sets the _xsrf cookie, POST /login sets the session cookie.
        Requests made with the cookie must echo _xsrf in the X-XSRFToken header."""
        r = self.http.get(f"{self.base}/login", timeout=60)
        r.raise_for_status()
        xsrf = self.http.cookies.get("_xsrf") or (re.search(r'name="_xsrf" value="([^"]+)"', r.text) or [None, ""])[1]
        r = self.http.post(f"{self.base}/login", data={"password": password, "_xsrf": xsrf}, allow_redirects=False, timeout=60)
        if r.status_code not in (302, 303) or "/login" in r.headers.get("Location", ""):
            raise SystemExit(f"login at {self.base}/login failed (HTTP {r.status_code}): wrong password, or the login box "
                             "belongs to the site's gateway rather than to Jupyter")
        self.http.headers["X-XSRFToken"] = str(self.http.cookies.get("_xsrf") or xsrf)
        if self.api("GET", "status").status_code != 200:
            raise SystemExit("logged in, but the API refuses the session")

    def _ws_auth(self) -> tuple[str, dict]:
        """Query string and connect options for the terminal websocket: the token, or the session cookie with the
        server's own origin (Jupyter refuses cross-origin websockets)."""
        origin = "/".join(self.base.split("/", 3)[:3])  # scheme://host
        if self.token:
            return f"?token={quote(self.token)}", {"header": [f"Authorization: token {self.token}"], "origin": origin}
        cookie = "; ".join(f"{c.name}={c.value}" for c in self.http.cookies)
        return "", {"cookie": cookie, "origin": origin, "header": [f"X-XSRFToken: {self.http.headers.get('X-XSRFToken', '')}"]}

    def api(self, method: str, path: str, **kw) -> requests.Response:
        for attempt in range(5):
            try:
                r = self.http.request(method, f"{self.base}/api/{path}", timeout=kw.pop("timeout", 120), **kw)
                if r.status_code in (502, 503, 504):
                    raise requests.ConnectionError(f"HTTP {r.status_code}")
                return r
            except (requests.ConnectionError, requests.Timeout) as exc:
                if attempt == 4:
                    raise
                print(f"  retry {path}: {exc}", file=sys.stderr)
                time.sleep(2 * (attempt + 1))
        raise RuntimeError("unreachable")

    # ------------------------------------------------------------------------------------ commands
    def run(self, cmd: str, timeout: float = 600.0, echo: bool = True) -> tuple[int, str]:
        """Run cmd with bash in a fresh terminal; returns (exit code, output). Output streams to stdout if echo."""
        r = self.api("POST", "terminals")
        r.raise_for_status()
        name = r.json()["name"]
        query, ws_opts = self._ws_auth()
        ws_url = re.sub(r"^http", "ws", self.base) + f"/terminals/websocket/{name}{query}"
        mark = uuid.uuid4().hex[:12]
        script = base64.b64encode(cmd.encode()).decode()
        line = (f"stty -echo; echo {script} | base64 -d > /dev/shm/jr_{mark}.sh; echo __B{mark}; "
                f"bash /dev/shm/jr_{mark}.sh; rc=$?; rm -f /dev/shm/jr_{mark}.sh; printf '\\n__E{mark} %s\\n' $rc\r")
        out: list[str] = []
        rc = None
        try:
            ws = websocket.create_connection(ws_url, timeout=30, **ws_opts)
            ws.send(json.dumps(["stdin", line]))
            # the end marker follows a newline, in case the output did not end with one: hold back blank lines
            # until something else follows, so that extra newline is not printed
            buf, started, t0, blanks = "", False, time.time(), 0
            ws.settimeout(5)
            while rc is None:
                if time.time() - t0 > timeout:
                    raise TimeoutError(f"no end after {timeout:.0f} s (the command is killed with its terminal; use nohup for long jobs)")
                try:
                    msg = json.loads(ws.recv())
                except websocket.WebSocketTimeoutException:
                    continue
                if msg[0] == "disconnect":
                    raise ConnectionError("terminal closed")
                if msg[0] != "stdout":
                    continue
                buf += ANSI.sub("", msg[1]).replace("\r\n", "\n").replace("\r", "\n")
                *lines, buf = buf.split("\n")
                for ln in lines:
                    if not started:
                        started = ln.strip() == f"__B{mark}"
                    elif ln.startswith(f"__E{mark} "):
                        rc = int(ln.split()[1])
                        break
                    elif not ln:
                        blanks += 1
                    else:
                        for x in [""] * blanks + [ln]:
                            out.append(x)
                            if echo:
                                print(x, flush=True)
                        blanks = 0
            ws.close()
        finally:
            self.api("DELETE", f"terminals/{name}")
        return rc, "\n".join(out)

    # ------------------------------------------------------------------------------------ files
    def size(self, path: str) -> int | None:
        r = self.api("GET", f"contents/{quote(path)}", params={"content": 0})
        return r.json().get("size") if r.status_code == 200 else None

    def mkdir(self, path: str) -> None:
        parts = [p for p in path.strip("/").split("/") if p]
        for i in range(1, len(parts) + 1):
            sub = "/".join(parts[:i])
            if self.size(sub) is None and self.api("GET", f"contents/{quote(sub)}", params={"content": 0}).status_code == 404:
                self.api("PUT", f"contents/{quote(sub)}", json={"type": "directory"}).raise_for_status()

    def put_file(self, local: Path, remote: str, offset: int, length: int, chunk: int) -> None:
        """Upload bytes [offset, offset + length) of local to remote, in chunks (1 = create, then append, -1 = last)."""
        name = posixpath.basename(remote)
        with local.open("rb") as f:
            f.seek(offset)
            n = max(1, -(-length // chunk))
            for k in range(1, n + 1):
                data = f.read(min(chunk, length - (k - 1) * chunk))
                body = {"type": "file", "format": "base64", "name": name, "path": remote,
                        "content": base64.b64encode(data).decode()}
                if n > 1:
                    body["chunk"] = -1 if k == n else k
                self.api("PUT", f"contents/{quote(remote)}", json=body, timeout=300).raise_for_status()

    def upload(self, local: Path, remote_dir: str, part_mb: int, chunk_mb: float, parallel: int) -> None:
        size = local.stat().st_size
        target = posixpath.join(remote_dir.strip("/"), local.name)
        print(f"sha256 of {local} ({size / 1e6:.0f} MB) ...", flush=True)
        h = hashlib.sha256()
        with local.open("rb") as f:
            for block in iter(lambda: f.read(1 << 24), b""):
                h.update(block)
        digest = h.hexdigest()
        part = part_mb << 20
        parts_dir = f"{target}.parts"
        self.mkdir(parts_dir)
        n = max(1, -(-size // part))
        todo = []
        for i in range(n):
            length = min(part, size - i * part)
            if self.size(f"{parts_dir}/{i:05d}") != length:
                todo.append((i, length))
        print(f"{n} parts of {part_mb} MB: {n - len(todo)} already on the server, {len(todo)} to send", flush=True)
        t0, sent = time.time(), 0

        def send(item: tuple[int, int]) -> int:
            i, length = item
            self.put_file(local, f"{parts_dir}/{i:05d}", i * part, length, int(chunk_mb * (1 << 20)))
            return length

        with ThreadPoolExecutor(parallel) as ex:
            for length in ex.map(send, todo):
                sent += length
                dt = time.time() - t0
                print(f"  {sent / 1e6:.0f}/{sum(x[1] for x in todo) / 1e6:.0f} MB, {sent / 1e6 / max(dt, 1e-6):.1f} MB/s", flush=True)
        q = lambda s: "'" + s.replace("'", "'\\''") + "'"  # noqa: E731
        rc, out = self.run(f"set -e; cat {q(parts_dir)}/* > {q(target)}.tmp && mv {q(target)}.tmp {q(target)} && "
                           f"sha256sum {q(target)} | cut -d' ' -f1", timeout=1800, echo=False)
        remote_digest = out.strip().splitlines()[-1] if out.strip() else ""
        if rc != 0 or remote_digest != digest:
            raise SystemExit(f"join failed or checksum differs (rc {rc}): local {digest}, server {remote_digest!r}; parts kept in {parts_dir}")
        self.run(f"rm -rf {q(parts_dir)}", echo=False)
        print(f"uploaded {target}: sha256 {digest} verified", flush=True)

    def download(self, remote: str, local: str) -> None:
        """local: a file name, or a folder (existing, or given with a trailing /) to download into."""
        dest = Path(local) / posixpath.basename(remote) if Path(local).is_dir() or local.endswith("/") else Path(local)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with self.http.get(f"{self.base}/files/{quote(remote)}", stream=True, timeout=300) as r:
            r.raise_for_status()
            tmp = dest.with_name(dest.name + ".part")
            with tmp.open("wb") as f:
                for block in r.iter_content(1 << 20):
                    f.write(block)
            tmp.replace(dest)
        print(f"downloaded {remote} -> {dest} ({dest.stat().st_size / 1e6:.1f} MB)")

    def ls(self, path: str) -> None:
        r = self.api("GET", f"contents/{quote(path.strip('/'))}")
        r.raise_for_status()
        for item in sorted(r.json().get("content") or [], key=lambda x: x["name"]):
            size = "" if item["type"] == "directory" else f"{(item.get('size') or 0) / 1e6:10.1f} MB"
            print(f"{item['last_modified'][:16]}  {size:>13}  {item['name']}{'/' if item['type'] == 'directory' else ''}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run"); p.add_argument("command"); p.add_argument("--timeout", type=float, default=600)
    p = sub.add_parser("upload"); p.add_argument("local", type=Path); p.add_argument("remote_dir")
    p.add_argument("--part-mb", type=int, default=64); p.add_argument("--chunk-mb", type=float, default=8)
    p.add_argument("--parallel", type=int, default=4)
    p = sub.add_parser("download"); p.add_argument("remote"); p.add_argument("local")
    p = sub.add_parser("ls"); p.add_argument("path", nargs="?", default="")
    args = ap.parse_args()
    j = Jupyter()
    if args.cmd == "run":
        rc, _ = j.run(args.command, timeout=args.timeout)
        sys.exit(rc)
    if args.cmd == "upload":
        j.upload(args.local, args.remote_dir, args.part_mb, args.chunk_mb, args.parallel)
    elif args.cmd == "download":
        j.download(args.remote, args.local)
    else:
        j.ls(args.path)


if __name__ == "__main__":
    main()
