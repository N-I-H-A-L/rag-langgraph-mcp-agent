"""MCP client: call the local web-search server and return the tool text.

LangGraph stays the client. The server is a streamable HTTP resource server,
so this helper starts it if needed and sends MCP_AUTH_TOKEN as a bearer token.
The graph node stays synchronous, so async MCP calls are wrapped here.
"""

import asyncio
import concurrent.futures
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

from langchain_mcp_adapters.client import MultiServerMCPClient

from mcp_auth import HOST, PORT, RESOURCE_SERVER_URL, load_token

SERVER_PATH = Path(__file__).resolve().parent / "mcp_search_server.py"

# Handle for the server process this client started. None until the first search.
# `subprocess.Popen | None` means the value is either a running process or nothing yet.
_server_proc: subprocess.Popen | None = None

# A lock lets only one thread run the start-up code at a time.
# web_search_sync can run in a worker thread, so two searches could otherwise
# both see a closed port and launch two servers.
_server_lock = threading.Lock()


def _port_is_open() -> bool:
    # Try a TCP connection to 127.0.0.1:8765.
    # Success means some process is already listening, so we should not start another.
    # OSError (connection refused) means nothing is listening yet.
    # `with` closes the test socket as soon as this block ends.
    try:
        with socket.create_connection((HOST, PORT), timeout=0.3): # Check if port is open that is, something is already listening on the port. If socket connection is established, it means server is already running. So return True.
            return True
        # Why use socket? Because client doesn't keep a permanent connection to the server. So it needs to check if the server is running by trying to connect to the port.
    except OSError:
        return False # Return false if server is not running.


def _ensure_server() -> None:
    """Start mcp_search_server.py unless something is already listening."""
    # `global` is required to assign the module-level _server_proc from in here.
    # Without it, `_server_proc = ...` would create a local variable and the
    # handle would be thrown away when this function returns.
    global _server_proc
    # `with` acquires the lock here and releases it on every return, including raises.
    # The port check, the launch, and the wait all stay inside so a second caller waits.
    with _server_lock:
        if _port_is_open(): # Check if server is already running. If true, return. Else we will start the server.
            return

        # load_token() reads .env into this process. The child inherits it.
        load_token()
        print(f"Starting MCP server at {RESOURCE_SERVER_URL}", file=sys.stderr)
        # Popen starts the server and returns immediately. It does not wait for
        # the process to finish, because the server is supposed to keep running.
        _server_proc = subprocess.Popen([sys.executable, str(SERVER_PATH)])

        # The process exists before it has opened the port, so wait until a
        # connection succeeds, the process dies, or 20 seconds pass.
        deadline = time.time() + 20
        while time.time() < deadline:
            if _port_is_open():
                return
            # poll() is None while the process is alive. A return code means it
            # exited first (for example, MCP_AUTH_TOKEN was missing).
            if _server_proc.poll() is not None:
                raise RuntimeError(
                    f"MCP server exited with status {_server_proc.returncode} "
                    "before it started listening."
                )
            time.sleep(0.2)
        raise RuntimeError(f"MCP server did not start listening on {HOST}:{PORT}.")
        # This waits until the newly started server actually opens the port (or dies, or the 20-second deadline expires).


def _server_config() -> dict:
    token = load_token()
    return {
        # streamable_http is the LangChain name for the SDK's streamable-http
        # transport. headers carries the bearer token the server's TokenVerifier
        # checks. A missing or wrong token is rejected before web_search runs.
        "search": {
            "transport": "streamable_http",
            "url": str(RESOURCE_SERVER_URL),
            "headers": {"Authorization": f"Bearer {token}"},
        }
    }


async def _web_search_async(query: str) -> str:
    _ensure_server()
    client = MultiServerMCPClient(_server_config())
    tools = await client.get_tools()  # LangChain tool objects from every connected server.
    tool = next((item for item in tools if item.name == "web_search"), None)
    if tool is None:
        names = [item.name for item in tools]
        raise RuntimeError(f"MCP server did not expose web_search. Tools: {names}")
    result = await tool.ainvoke({"query": query})
    return _tool_text(result)


# Normalize whatever the tool returns into one string.
def _tool_text(result) -> str:
    if isinstance(result, str):
        return result
    if isinstance(result, list):
        parts = []
        for item in result:
            if isinstance(item, dict) and item.get("text"):
                parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    if isinstance(result, dict) and result.get("text"):
        return str(result["text"])
    return str(result)


def _run_async(coro):
    # app.invoke and client.evaluate are sync. If no event loop is running,
    # asyncio.run is enough. If one is running, run the coroutine in a thread.
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


# Graph uses sync app.invoke, so the node calls this wrapper, not the coroutine.
def web_search_sync(query: str) -> str:
    return _run_async(_web_search_async(query))
