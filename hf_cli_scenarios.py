import asyncio
import json
import os
from openai_agents import Agent, tool, run

HF_TIMEOUT = 30
SESSION_FILE = "session.json"

class HFCLIExecutor(Agent):
    name = "HF-CLI-EXECUTOR"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.memory.setdefault("hf_token", None)
        self._load_session()

    def _load_session(self):
        if os.path.exists(SESSION_FILE):
            with open(SESSION_FILE, "r") as f:
                self.memory.update(json.load(f))

    def _save_session(self):
        with open(SESSION_FILE, "w") as f:
            json.dump(self.memory, f)

    @tool(name="run_hf_cli", description="Run huggingface-cli commands")
    async def run_hf_cli(self, cmd: str, args: list[str] = None) -> dict:
        args = args or []
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
        stdout_str = stdout.decode()
        parsed = stdout_str.strip()
        if "{" in stdout_str and "}" in stdout_str:
            try:
                parsed = json.loads(stdout_str[stdout_str.find("{"): stdout_str.rfind("}") + 1])
            except json.JSONDecodeError:
                pass
        result = {
            "exit": proc.returncode,
            "stdout": parsed,
            "stderr": stderr.decode().strip(),
        }
        return result

    async def ensure_login(self, token: str):
        if self.memory.get("hf_token") == token:
            return
        await self.run_hf_cli("login", ["--token", token])
        self.memory["hf_token"] = token
        self._save_session()

async def cli_chat(token: str):
    agent = HFCLIExecutor()
    await agent.ensure_login(token)
    res = await agent.run_hf_cli("whoami")
    print(res)

async def full_mcp(token: str):
    agent = HFCLIExecutor()
    await agent.ensure_login(token)
    tasks = [
        agent.run_hf_cli("models", ["list", "--limit", "2", "--json"]),
        agent.run_hf_cli("datasets", ["list", "--limit", "2", "--json"]),
    ]
    results = await asyncio.gather(*tasks)
    for r in results:
        print(r)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python hf_cli_scenarios.py <HF_TOKEN>")
        sys.exit(1)
    token = sys.argv[1]
    asyncio.run(cli_chat(token))
    asyncio.run(full_mcp(token))
