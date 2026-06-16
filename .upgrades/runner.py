"""Scheduled-upgrade runner.

Applies the *next* not-yet-applied item from the backlog (``tasks.py``) and
commits it to the repo via the GitHub Contents API. Commits made through that
API are signed by GitHub (so they show **Verified**) and are attributed to the
owner of the token in ``UPGRADE_TOKEN`` -- i.e. you.

Design constraints that keep this cheap and safe:

* **Zero LLM tokens.** Every change is deterministic Python; no model calls.
* **One file per run.** The Contents API commits a single file, which is exactly
  what guarantees the Verified signature.
* **No state file.** Each task declares ``applied(root)``; the runner picks the
  first task that is not yet applied, so the backlog is idempotent and finite --
  once everything is applied, runs become no-ops.
* **Stdlib only.** Uses ``urllib`` so CI needs no ``pip install``.
"""

from __future__ import annotations

import base64
import json
import os
import sys
import urllib.error
import urllib.request
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tasks import UPGRADES  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
API = "https://api.github.com"


def _request(method: str, url: str, token: str, payload: Optional[dict] = None) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("X-GitHub-Api-Version", "2022-11-28")
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req) as resp:
        body = resp.read().decode()
        return json.loads(body) if body else {}


def _get_sha(repo: str, path: str, token: str, branch: str) -> Optional[str]:
    """Return the current blob SHA of ``path`` on ``branch``, or None if absent."""

    url = f"{API}/repos/{repo}/contents/{path}?ref={branch}"
    try:
        return _request("GET", url, token).get("sha")
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise


def _put_file(
    repo: str, path: str, content: str, message: str, token: str, branch: str
) -> dict:
    """Create or update ``path`` via the Contents API (yields a Verified commit)."""

    sha = _get_sha(repo, path, token, branch)
    payload = {
        "message": message,
        "content": base64.b64encode(content.encode()).decode(),
        "branch": branch,
    }
    if sha:
        payload["sha"] = sha
    url = f"{API}/repos/{repo}/contents/{path}"
    return _request("PUT", url, token, payload)


def main() -> int:
    token = os.environ.get("UPGRADE_TOKEN") or os.environ.get("GH_TOKEN")
    repo = os.environ.get("GITHUB_REPOSITORY", "sudhanshu-shivam-dev/agent-harness-from-scratch")
    branch = os.environ.get("UPGRADE_BRANCH", "main")

    if not token:
        print("No UPGRADE_TOKEN set; nothing to do (configure the repo secret).")
        return 0

    for upgrade in UPGRADES:
        if upgrade["applied"](ROOT):
            continue
        path = upgrade["path"]
        content = upgrade["render"](ROOT)
        message = upgrade["message"]
        result = _put_file(repo, path, content, message, token, branch)
        commit = result.get("commit", {})
        print(f"Applied upgrade '{upgrade['id']}' -> {path}")
        print(f"  commit: {commit.get('sha', '?')[:10]}  message: {message}")
        return 0

    print("Backlog fully applied; nothing to do.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
