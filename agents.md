# Integration Guide – **openai‑agents‑python** with **Hugging Face CLI**

This guide demonstrates idiomatic patterns for invoking `huggingface-cli` commands *from* an **OpenAI Agents SDK** tool, including best practices for streaming, error handling, sandboxing, and leveraging Codex‑like code‑execution helpers.

---

## 1. Quick Start Snippet

```python
from agents import Agent
from agents.tool import function_tool
from agents.run import Runner
import asyncio, json, textwrap

HF_TIMEOUT = 30  # seconds

class HFCLIExecutor(Agent):
    name = "AG-EXEC-CLI"

    @function_tool(name="run_hf_cli", description="Run huggingface-cli commands")
    async def run_hf_cli(self, cmd: str, args: list[str] = []) -> dict:
        """Execute huggingface‑cli <cmd> <args> and return structured result"""
        proc = await asyncio.create_subprocess_exec(
            "huggingface-cli", cmd, *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), HF_TIMEOUT)
        except asyncio.TimeoutError:
            proc.kill()
            return {"exit": -1, "error": "timeout"}
        exit_code = proc.returncode
        # Attempt to parse JSON block if present
        stdout_str = stdout.decode()
        if "{" in stdout_str and "}" in stdout_str:
            try:
                json_block = json.loads(stdout_str[stdout_str.find("{"): stdout_str.rfind("}") + 1])
                parsed = json_block
            except json.JSONDecodeError:
                parsed = stdout_str.strip()
        else:
            parsed = stdout_str.strip()
        return {
            "exit": exit_code,
            "stdout": parsed,
            "stderr": stderr.decode().strip(),
        }

if __name__ == "__main__":
    # ad‑hoc run for local dev
    print(asyncio.run(Runner.run(agent=HFCLIExecutor(), input="run_hf_cli cmd='whoami'")))
```

### Why this pattern?

* **asyncio.create\_subprocess\_exec** avoids blocking the event loop used by other agents.
* **wait\_for** enforces a hard timeout to keep MCP requests bounded.
* Outputs are *attempt‑parsed* to JSON if the CLI supports `--json` (e.g., `models list --json`).

---

## 2. Hugging Face CLI JSON‑Friendly Flags

| Command         | Recommended Flags                        | Returns JSON?   |         |
| --------------- | ---------------------------------------- | --------------- | ------- |
| `models list`   | `--author <org> --sort downloads --json` | ✅               |         |
| `datasets list` | `--json`                                 | ✅               |         |
| `spaces list`   | `--json`                                 | ✅               |         |
| `whoami`        | *(none)*                                 | ❌ (parse stderr | stdout) |

Always pass `--json` when available to make parsing trivial.

---

## 3. Codex Tooling Synergy

The **OpenAI Codex Playground** or GPT‑4o‑code model can auto‑generate wrappers for new CLI commands:

```python
# in a dev notebook
from openai import OpenAI
client = OpenAI()

generated = client.chat.completions.create(
    model="gpt-4o-code",  # Codex style
    messages=[
        {"role": "user", "content": "Write an asyncio Python wrapper for 'huggingface-cli datasets list --json'"}
    ],
)
print(generated.choices[0].message.content)
```

**Tip**: integrate this generation step into `AG-CORE-ORCH` to auto‑scaffold wrappers for infrequent commands.

---

## 4. Authentication Lifecycle Pattern

```python
class HFAuthMixin:
    async def ensure_login(self, token: str):
        # cache token in Redis or Agent memory
        cached = self.memory.get("hf_token")
        if cached == token:
            return
        await self.run_hf_cli("login", ["--token", token])
        self.memory["hf_token"] = token
```

Call `ensure_login()` before executing any command requiring auth.

---

## 5. Streaming Large Outputs

For listings returning thousands of models, stream lines to the bus:

```python
async for line in proc.stdout:
    await ctx.emit("stream", line.decode())
```

Down‑stream agent (`AG-OBS-MON`) or the gateway can chunk‑forward to HTTP clients (Server‑Sent Events or chunked‑transfer).

---

## 6. Error Classification Matrix

| Exit Code | Likely Cause                   | Retry?        |
| --------- | ------------------------------ | ------------- |
| `0`       | success                        | ─             |
| `1`       | bad flag / auth error          | ❌             |
| `126/127` | command not found / permission | ❌ (ops alert) |
| `-1`      | timeout (killed)               | ✅ once        |

Map these statuses in `AG-CORE-ORCH` to the Failure & Retry Policy (see `agents.md`).

---

## 7. Embedding CLI Logic as Tools vs. External Subprocess

* **Pros of subprocess**: decoupled from Python API stability; mirrors real user workflows.
* **Cons**: slower than `huggingface_hub` SDK for bulk fetches; parsing required.

*Recommendation*: keep CLI path for *agentic text* fidelity but optionally fall back to direct SDK calls where performance critical (future optimisation ticket).

---

## 8. Unit Test Template (pytest‑asyncio)

```python
import pytest, asyncio
from agents.exec_cli import HFCLIExecutor

@pytest.mark.asyncio
async def test_models_list_json():
    agent = HFCLIExecutor()
    out = await agent.run_hf_cli("models", ["list", "--limit", "1", "--json"])
    assert out["exit"] == 0
    assert isinstance(out["stdout"], list)
```

---

*Last updated: 2025‑06‑28*
