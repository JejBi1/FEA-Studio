"""pytest setup shared by all tests.

The MCP server exposes only the `core` tool group by default (fea_core/mcp/groups.py); the tests use every tool, so they start
the server with all groups. Tests of the default / of the groups pass `groups_spec=` explicitly.
"""
import os

os.environ.setdefault("FEA_MCP_GROUPS", "all")
os.environ.setdefault("FEA_I18N_CACHE", "0")      # persisted start-up translations (fea_core.i18n) off – tests see the catalogue
