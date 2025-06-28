# HF CLI Chat and MCP Scenario

This document describes an agentic workflow for interacting with the Hugging Face CLI using the patterns from `agents.md`. It consists of two stages:

1. **CLI Chat** – a single agent (`HFCLIExecutor`) logs in and performs a small command like `whoami`.
2. **Full MCP** – the same agent runs multiple commands (`models list` and `datasets list`) in parallel.
3. **Chat Delegation** – a `HFChatAgent` converses with the user and hands off requests to `HFCLIExecutor`.

## Agent Implementation

```python
# hf_cli_scenarios.py (excerpt)
class HFCLIExecutor(Agent):
    name = "HF-CLI-EXECUTOR"

    def __init__(self, *args, **kwargs):
        super().__init__(name=self.name, tools=[], *args, **kwargs)
        self.session = {}
        self.session.setdefault("hf_token", None)
        self._load_session()

    def _load_session(self):
        if os.path.exists(SESSION_FILE):
            with open(SESSION_FILE, "r") as f:
                self.session.update(json.load(f))

    def _save_session(self):
        with open(SESSION_FILE, "w") as f:
            json.dump(self.session, f)
```

`_load_session()` and `_save_session()` keep the HF token and other memory across runs.

```python
@tool(name="run_hf_cli", description="Run huggingface-cli commands")
async def run_hf_cli(self, cmd: str, args: list[str] = None) -> dict:
    proc = await asyncio.create_subprocess_exec(
        "huggingface-cli", cmd, *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    ...  # parse stdout/stderr as shown in agents.md
```

The `cli_chat()` coroutine logs in (if needed), runs `whoami`, and prints the result. `full_mcp()` triggers several CLI commands concurrently. `chat_with_agents()` demonstrates delegating user requests from `HFChatAgent` to `HFCLIExecutor` via a handoff. See `hf_cli_scenarios.py` for details.

## Usage

```bash
python hf_cli_scenarios.py <HF_TOKEN>
```

On the first run, the token is saved to `session.json`. Subsequent runs reuse the token without prompting for login. The script prints results for `cli_chat`, `full_mcp`, and a final `chat_with_agents` conversation.
