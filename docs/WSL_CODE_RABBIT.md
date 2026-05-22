# WSL / CodeRabbit Troubleshooting

## Root Cause

CodeRabbit CLI for Windows workflows runs in WSL. If a `coderabbit` file in
`C:\Users\yanbo\.local\bin` starts with `7F 45 4C 46`, it is an ELF Linux
binary. Do not double-click it in Windows Explorer and do not open it with
Notepad or VS Code.

## PowerShell vs WSL Commands

PowerShell:

```powershell
Remove-Item -LiteralPath "C:\Users\yanbo\.local\bin\coderabbit" -Force
Remove-Item -LiteralPath "C:\Users\yanbo\.local\bin\cr" -Force
```

WSL bash:

```bash
rm -f /mnt/c/Users/yanbo/.local/bin/coderabbit
rm -f /mnt/c/Users/yanbo/.local/bin/cr
```

Do not run `Remove-Item` inside WSL. It is a PowerShell command.

## WSL Python Setup

From WSL:

```bash
cd /mnt/c/Users/yanbo/Desktop/agent/ev-ner-agent
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
python -m task_agent.main --health
pytest -q
```

If `python` is missing before activation, use `python3`. Inside the activated
virtual environment, `python` should resolve to `.venv/bin/python`.

## Proxy Failure

This error means WSL is configured to use a proxy that is not reachable:

```text
curl: (7) Failed to connect to 10.255.255.254 port 7897
```

Temporarily clear proxy variables in WSL:

```bash
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
```

Then retry:

```bash
curl -fsSL https://cli.coderabbit.ai/install.sh | sh
```

If you need a proxy, set it to a host/port reachable from WSL.

## CodeRabbit Install In WSL

```bash
curl -fsSL https://cli.coderabbit.ai/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
coderabbit auth login --agent
coderabbit review --agent -t uncommitted
```

If CodeRabbit remains unavailable, use the local quality gate:

```bash
python -m task_agent.main --health
pytest -q
```
