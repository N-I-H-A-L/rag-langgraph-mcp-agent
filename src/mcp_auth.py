"""Shared auth settings for the local web-search MCP server and client.

The MCP Python SDK authenticates streamable HTTP servers with a bearer token.
stdio has no header to carry one, so both processes agree on the URL, the
scope, and the MCP_AUTH_TOKEN environment variable.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import AnyHttpUrl

HOST = "127.0.0.1"
PORT = 8765
MCP_PATH = "/mcp"
# A valid token must carry the scope "user" to access the MCP server. Else it will throw 403.
SCOPE = "user"

RESOURCE_SERVER_URL = AnyHttpUrl(f"http://{HOST}:{PORT}{MCP_PATH}")
# Advertised in protected-resource metadata. This app has no separate
# authorization server; EnvTokenVerifier checks the shared token itself.
ISSUER_URL = AnyHttpUrl(f"http://{HOST}:{PORT}")

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
TOKEN_ENV = "MCP_AUTH_TOKEN"


def load_token() -> str:
    load_dotenv(ENV_PATH)
    token = os.getenv(TOKEN_ENV, "").strip()
    if not token:
        raise ValueError(f"{TOKEN_ENV} is missing. Add it to your .env file.")
    return token
