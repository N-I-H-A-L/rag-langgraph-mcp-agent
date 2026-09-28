"""MCP client: start the local web-search server and call web_search.

LangGraph stays the client. This helper talks to mcp_search_server.py over
stdio and returns the tool's text. The graph node stays synchronous, so
async MCP calls are wrapped here.
"""

import asyncio
import concurrent.futures
import sys
from pathlib import Path

from langchain_mcp_adapters.client import MultiServerMCPClient

SERVER_PATH = Path(__file__).resolve().parent / "mcp_search_server.py"


def _server_config() -> dict:
    return {
        # "search" is the name of the MCP server we are calling.
        # "command" is the path to the Python interpreter. (which will be "python" in most cases)
        # "args" is the list of arguments to pass to the Python interpreter. (which will be [str(SERVER_PATH)], that is, the path to the MCP server)
        # So overall the command is: python mcp_search_server.py (that is, starting the MCP server)
        # "transport" is the transport protocol to use for the MCP server. (which will be "stdio" in most cases)
        "search": {
            "command": sys.executable,
            "args": [str(SERVER_PATH)], # str(SERVER_PATH) is the path to the MCP server
            "transport": "stdio",
        }
    }

# Using a coroutine to call the MCP server.
async def _web_search_async(query: str) -> str:
    client = MultiServerMCPClient(_server_config()) # It starts the MCP server from the config.
    tools = await client.get_tools() # It gets the tools of the servers. It will return LangChain tool objects.
    # Suppose if there are multiple servers, tools will be a list of tool objects exposed from all the servers in a flat list.
    tool = next((item for item in tools if item.name == "web_search"), None) # next() is a function that returns the first item in the list that satisfies the condition. In this case, it returns the web_search tool.
    if tool is None:
        names = [item.name for item in tools]
        raise RuntimeError(f"MCP server did not expose web_search. Tools: {names}")
    result = await tool.ainvoke({"query": query}) # ainvoke is an asynchronous version of invoke. It returns a coroutine.
    return _tool_text(result)

# To normalize the result, we convert the result to a string.
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

# Helper method to run the coroutine.
def _run_async(coro):
    # app.invoke and client.evaluate are sync. If no event loop is running,
    # asyncio.run is enough. If one is running, run the coroutine in a thread.
    try:
        asyncio.get_running_loop()
        # Check if an event loop is running. If not, run the coroutine using asyncio.run.
        # If an event loop is running, run the coroutine in a thread.
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool: # It creates a thread pool with a maximum of 1 worker.
        return pool.submit(asyncio.run, coro).result()

# To make the web_search_sync function synchronous, we wrap the async _web_search_async function in a synchronous function. As MCP I/O is asynchronous.
# Graph uses sync app.invoke, so we do not call this function directly from the node. web_search_sync wraps it.
def web_search_sync(query: str) -> str:
    return _run_async(_web_search_async(query))
