"""MCP server: one tool, web_search, backed by DuckDuckGo.

This file is a separate process. The LangGraph app (the MCP client) starts it
over stdio and calls the tool by name. Do not print to stdout — stdout is the
MCP protocol. Errors go to stderr.
"""

import sys

from ddgs import DDGS
from ddgs.exceptions import DDGSException
from mcp.server.fastmcp import FastMCP

# FastMCP is a class that creates an MCP server. It is a subclass of MCP.
# "mcp" is an instance of FastMCP.
# "web-search" is the name of the server instance we created.
mcp = FastMCP("web-search")

MAX_RESULTS = 5

# @mcp.tool() is a decorator that registers the function as a tool that can be called by the client. Without decorator, you would write something like `mcp.add_tool(web_search)` but this decorator does that automatically for you. You can inspect it to know more.
@mcp.tool()
def web_search(query: str) -> str:
    """Search the public web and return titles, snippets, and URLs."""

    # Try catch block
    try:
        hits = list(DDGS().text(query, max_results=MAX_RESULTS))
        # DDGS() creates a DuckDuckGo client. It will search the web and return the results and list() converts the generator to a list.
    except (DDGSException, OSError, RuntimeError) as exc: # If one of these exceptions are raised
        print(f"web_search failed: {exc}", file=sys.stderr)
        return f"Web search failed: {exc}"

    # If no hits are found, return a message
    if not hits:
        return "No web results found."

    lines = []
    for i, hit in enumerate(hits, start=1):
        title = hit.get("title") or ""
        body = hit.get("body") or hit.get("description") or ""
        href = hit.get("href") or hit.get("url") or ""
        lines.append(f"{i}. {title}\n{body}\n{href}")
    return "\n\n".join(lines) # Return the results as a string by joining the lines with double newlines. Since when passing {context} to the prompts in graph.py, it expects a string.


if __name__ == "__main__":
    # Run the MCP server using stdio transport
    # Until run, mcp is just an object with a tool list. After run, it becomes an MCP server that can be called by the client.
    mcp.run(transport="stdio")
