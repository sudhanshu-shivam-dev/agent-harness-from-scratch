# .upgrades — scheduled incremental improvements

A small, deterministic system that lands **one real improvement at a time** on a
schedule, with commits that are **Verified** and **attributed to you** — at
**zero LLM-token cost**.

## How it works

- `tasks.py` holds an ordered **backlog** of single-file improvements. Each item
  declares whether it is already `applied` and how to `render` its file.
- `runner.py` picks the first not-yet-applied item, computes the file contents,
  and commits it through the GitHub **Contents API**. API commits are signed by
  GitHub (so they show **Verified**) and authored by the owner of the token.
- `.github/workflows/scheduled-upgrade.yml` runs the runner on a cron (every
  ~6 hours) and on manual dispatch.

Because each item is idempotent and the runner stops once everything is applied,
the backlog is **finite**: it improves the repo steadily, then goes quiet. Add
more items to `tasks.py` whenever you want it to keep going.

## One-time setup

1. Create a **fine-grained personal access token** (GitHub → Settings →
   Developer settings → Fine-grained tokens):
   - Repository access: only `agent-harness-from-scratch`.
   - Permissions: **Contents → Read and write**.
2. Add it as a repo secret named `UPGRADE_TOKEN`
   (repo → Settings → Secrets and variables → Actions → New repository secret).
3. (Optional) Trigger a manual run from the **Actions** tab to verify it works.

## Adding your own upgrades

Append a dict to `UPGRADES` in `tasks.py`. Keep each one scoped to a **single
file** — that is what keeps the resulting commit Verified.
