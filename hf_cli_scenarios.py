import asyncio
import json
import os
from agents import Agent, handoff
from agents.tool import function_tool
from agents.run import Runner

from agents.repl import run_demo_loop

HF_TIMEOUT = 30
SESSION_FILE = "session.json"

class HFCLIExecutor(Agent):
    name = "HF-CLI-EXECUTOR"

    def __init__(self, *args, **kwargs):
        super().__init__(name=self.name, tools=[], *args, **kwargs)
        self.session = {}
        self.session.setdefault("hf_token", None)
        self._load_session()
        self.tools.append(function_tool(self.run_hf_cli, name_override="run_hf_cli", description_override="Run huggingface-cli commands"))

    def _load_session(self):
        if os.path.exists(SESSION_FILE):
            with open(SESSION_FILE, "r") as f:
                self.session.update(json.load(f))

    def _save_session(self):
        with open(SESSION_FILE, "w") as f:
            json.dump(self.session, f)

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

    @function_tool
    async def list_models(self, author: str | None = None, limit: int = 10) -> list:
        """List models via huggingface-cli with JSON output"""
        args = ["list", "--limit", str(limit), "--json"]
        if author:
            args.extend(["--author", author])
        result = await self.run_hf_cli("models", args)
        return result["stdout"]

    @function_tool
    async def list_datasets(self, limit: int = 10) -> list:
        """List datasets via huggingface-cli with JSON output"""
        args = ["list", "--limit", str(limit), "--json"]
        result = await self.run_hf_cli("datasets", args)
        return result["stdout"]

    async def ensure_login(self, token: str):
        if self.session.get("hf_token") == token:
            return
        await self.run_hf_cli("login", ["--token", token])
        self.session["hf_token"] = token
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

class HFChatAgent(Agent):
    """Agent that chats with the user and delegates HF CLI work."""

    name = "HF-CHAT-AGENT"

    def __init__(self, hf_agent: HFCLIExecutor, *args, **kwargs):
        self.hf_agent = hf_agent
        super().__init__(
            name=self.name,
            handoffs=[
                handoff(
                    self.hf_agent,
                    tool_name_override="hf_cli_agent",
                    tool_description_override="Delegate CLI requests to HF agent",
                )
            ],
            *args,
            **kwargs,
        )

async def chat_with_agents(token: str, message: str):
    hf_agent = HFCLIExecutor()
    await hf_agent.ensure_login(token)
    chat_agent = HFChatAgent(hf_agent)
    result = await Runner.run(chat_agent, message)
    print(result)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python hf_cli_scenarios.py <HF_TOKEN> [chat]")
        sys.exit(1)
    token = sys.argv[1]
    if len(sys.argv) > 2 and sys.argv[2] == "chat":
        async def interactive():
            hf_agent = HFCLIExecutor()
            await hf_agent.ensure_login(token)
            chat_agent = HFChatAgent(hf_agent)

            print("Starting interactive chat. Press Ctrl+C to exit.")
            await run_demo_loop(chat_agent, stream=False)

        asyncio.run(interactive())
    else:
        asyncio.run(cli_chat(token))
        asyncio.run(full_mcp(token))
        asyncio.run(chat_with_agents(token, "List two models"))
