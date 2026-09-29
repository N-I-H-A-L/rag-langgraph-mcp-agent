"""MCP server: one tool, web_search, backed by DuckDuckGo.

This process is an HTTP resource server. The LangGraph client connects to
http://127.0.0.1:8765/mcp and must send the bearer token from MCP_AUTH_TOKEN.
Calls without that token are rejected.
"""

import hmac
import sys

from ddgs import DDGS
from ddgs.exceptions import DDGSException
from mcp.server.auth.provider import AccessToken, TokenVerifier
from mcp.server.auth.settings import AuthSettings
from mcp.server.fastmcp import FastMCP

from mcp_auth import HOST, ISSUER_URL, PORT, RESOURCE_SERVER_URL, SCOPE, load_token


# TokenVerifier is the SDK hook for resource-server auth. FastMCP calls
# verify_token on every request that presents an Authorization: Bearer header.
class EnvTokenVerifier(TokenVerifier):
    """Accept only the shared MCP_AUTH_TOKEN, and only for this server's URL."""

    # This method runs on every request that presents an Authorization: Bearer header.
    async def verify_token(self, token: str) -> AccessToken | None:
        try:
            expected = load_token() # load the token from the environment variable
        except ValueError:
            return None
        if not token or not hmac.compare_digest(token.encode(), expected.encode()): # compare the token with the expected token
            return None
        # resource must equal AuthSettings.resource_server_url when
        # validate_token_resource is on, or the SDK rejects the token.
        return AccessToken(
            token=token,
            client_id="rag-agent",
            scopes=[SCOPE], # return an access token with the scope "user"
            resource=str(RESOURCE_SERVER_URL), # return the resource server url
        )


# FastMCP is the MCP server. "web-search" is its name.
# token_verifier and auth turn on bearer-token checks for streamable HTTP.
# stdio cannot do this: it has no Authorization header.
mcp = FastMCP(
    "web-search",
    json_response=True,
    host=HOST,
    port=PORT,
    log_level="WARNING",
    token_verifier=EnvTokenVerifier(),
    auth=AuthSettings(
        issuer_url=ISSUER_URL,
        resource_server_url=RESOURCE_SERVER_URL,
        required_scopes=[SCOPE], # required scopes for the access token
        validate_token_resource=True, # AccessToken.resource must equal AuthSettings.resource_server_url when validate_token_resource is on, or the SDK rejects the token.
    ),
)

MAX_RESULTS = 5

# @mcp.tool() registers the function as a tool the client can call.
@mcp.tool()
def web_search(query: str) -> str:
    """Search the public web and return titles, snippets, and URLs."""

    try:
        hits = list(DDGS().text(query, max_results=MAX_RESULTS))
        # DDGS() creates a DuckDuckGo client. list() converts the generator to a list.
    except (DDGSException, OSError, RuntimeError) as exc:
        print(f"web_search failed: {exc}", file=sys.stderr)
        return f"Web search failed: {exc}"

    if not hits:
        return "No web results found."

    lines = []
    for i, hit in enumerate(hits, start=1):
        title = hit.get("title") or ""
        body = hit.get("body") or hit.get("description") or ""
        href = hit.get("href") or hit.get("url") or ""
        lines.append(f"{i}. {title}\n{body}\n{href}")
    # graph.py puts this string into {context}, so the tool returns text, not a list.
    return "\n\n".join(lines)


if __name__ == "__main__":
    try:
        load_token() # load the token from the environment variable
    except ValueError as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1) from exc
    mcp.run(transport="streamable-http") # run the MCP server using the streamable-http transport
