# EV Task Agent Launch Checklist

Run these commands from `ev-ner-agent` before demo, handoff, or deployment:

```powershell
python -m task_agent.main --launch-check
```

`--launch-check` returns `ok: true`, `summary.total == 10`, and exit code `0`.
If any gate returns `ok: false`, the CLI exits with code `1` so CI/deploy scripts can fail fast.
It runs pytest, health check, 6-flow smoke, compatibility route check, dependency metadata check, compile check, wheel build check, installed-wheel smoke, Ruff static check, and default DB pollution guard in one command.

After `pip install -e ".[dev]"` or installing the built wheel, these console commands must be available:

- `ev-task-agent --health`
- `ev-task-agent --smoke`
- `ev-task-mcp`

If you need to run each gate manually:

```powershell
pytest -q
python -m task_agent.main --health
python -m task_agent.main --smoke
ev-task-agent --health
python -m compileall -q task_agent cs_agent run.py
python -m ruff check task_agent tests\task_agent
Test-Path -LiteralPath 'cs_agent\knowledge\task_agent.db'
```

Expected results:

- `pytest -q` passes all tests.
- `--health` returns `ok: true`.
- `ev-task-agent --health` returns `ok: true` after package installation.
- `--smoke` returns `ok: true` and `summary.total == 6`.
- `compileall` exits with code `0`.
- `ruff` returns `All checks passed!`.
- `Test-Path` returns `False`; diagnostics and smoke should not create the default repo-local task DB.

Current deployable task flows:

- Purchase recommendation and test-drive booking.
- Aftersales diagnosis and service appointment.
- Roadside assistance dispatch.
- Long-distance charging plan.
- Insurance claim assistance.
- Complaint escalation.
