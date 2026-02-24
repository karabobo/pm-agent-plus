# Repository Guidelines

## Project Structure & Module Organization
- `src/main.py` is the runtime entrypoint.
- Core code lives in `src/pm_agent/`, organized by domain (`ai/`, `strategy/`, `execution/`, `polymarket/`, `server/`, `db/`, etc.).
- `frontend/dist/` contains built dashboard assets served by FastAPI in `src/pm_agent/server/__init__.py`.
- Root scripts `start.sh`, `status.sh`, and `stop.sh` manage background runs.
- Runtime artifacts (`pm-agent-plus.log`, `pm-agent-plus.pid`, `*.db`) are local outputs and should not be committed.

## Build, Test, and Development Commands
```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
cd src && PYTHONPATH=. python3.11 main.py
```
Sets up a local environment and runs the agent in foreground mode.

```bash
./start.sh
./status.sh
./stop.sh
```
Starts, checks, and stops the background process (`pm-agent-plus`) from repo root.

## Coding Style & Naming Conventions
- Follow existing Python style: 4-space indentation, PEP 8 spacing, and readable, small functions.
- Use `snake_case` for files/functions/variables, `PascalCase` for classes, and `UPPER_SNAKE_CASE` for constants.
- Keep new modules in the matching domain folder under `src/pm_agent/`.
- Prefer explicit type hints and dataclasses for shared structured state.
- No formatter/linter config is committed; keep formatting consistent with nearby code when editing.

## Testing Guidelines
- No formal test suite is currently committed. Add tests under `tests/` using `pytest` with names like `test_risk.py`.
- For behavior changes, include manual verification steps in the PR (for example: startup succeeds, one decision cycle runs, `/api/stats` responds).
- For trading/risk logic, prioritize deterministic unit tests before merge.

## Commit & Pull Request Guidelines
- Use Conventional Commit-style prefixes used in history (`feat:`, `fix:`, `chore:`).
- Keep each commit focused on one logical change.
- PRs should include:
  - Clear summary and scope
  - Linked issue/ticket (if available)
  - Test evidence (commands run or manual checks)
  - Screenshots for dashboard/UI changes

## Security & Configuration Tips
- Never commit `.env`, API keys, or private keys; start from `.env.example`.
- Develop with `SIMULATION_MODE=true` before any live-wallet configuration.
- Treat local DB files and logs as sensitive operational data.
