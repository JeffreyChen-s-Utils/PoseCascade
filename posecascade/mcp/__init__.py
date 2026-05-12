"""Model Context Protocol server for PoseCascade.

Exposes a small set of headless, read-only tools an LLM agent can call to
introspect models, validate declarative animation documents, and bench
the cloth solver. The server runs in a subprocess over stdio — see
``posecascade.mcp.server`` for the entry point and ``docs/mcp.md`` for
how to register it with any MCP-aware client.
"""
