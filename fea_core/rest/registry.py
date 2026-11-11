"""Collection of the tools of the MCP server for the REST API (without MCPServer, without the MCP protocol).

The modules `fea_core.mcp.tools_*` register the tools through `reg.tool(**kw)` – here the functions are only collected.
The parameter schemas and the validation of the arguments are taken from the same SDK code as in the MCP server
(`Tool.from_function` → pydantic model of the arguments), so REST and MCP have the same names, descriptions and schemas.
Resources and prompts are ignored.
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable

from ..mcp.session import Session


@dataclass
class RestTool:
    name: str
    description: str
    fn: Callable[..., Any]
    parameters: dict              # JSON schema of the arguments (the same as the MCP inputSchema)
    meta: Any                     # FuncMetadata SDK – validate_arguments()

    def validate(self, args: dict) -> dict:
        """Validates and converts the arguments (pydantic) → kwargs of the function. Unknown keys = error (a typo in the REST call)."""
        known = set(self.meta.arg_model.model_fields)
        known |= {f.alias for f in self.meta.arg_model.model_fields.values() if f.alias}
        bad = sorted(set(args) - known)
        if bad:
            raise ValueError(f"unknown {', '.join(bad)} (allowed: {', '.join(sorted(known)) or 'none'})")
        return self.meta.validate_arguments(args)

    def info(self) -> dict:
        return {"name": self.name, "description": self.description, "parameters": self.parameters}


class _Collector:
    """Replacement of `_Registrar` from `fea_core.mcp.server`: tool() only stores the function, resource / prompt do nothing."""

    def __init__(self):
        self.tools: dict[str, RestTool] = {}

    def tool(self, **kw):
        def deco(fn):
            from mcp.server.mcpserver.tools.base import Tool
            from ..mcp.server import forbid_extra
            name = kw.get("name") or fn.__name__
            desc = kw.get("description") or inspect.getdoc(fn) or ""
            t = Tool.from_function(fn, name=name, description=desc,
                                   structured_output=kw.get("structured_output"))
            forbid_extra(t)              # schema as in MCP (additionalProperties: false)
            self.tools[name] = RestTool(name, desc, fn, t.parameters, t.fn_metadata)
            return fn
        return deco

    def _ignore(self, *a, **kw):
        return lambda fn: fn

    resource = prompt = _ignore


def tool_modules() -> list:
    """The same modules and order as `fea_core.mcp.server.build_server` (REST does not expose the prompts)."""
    from ..mcp.server import tool_modules as mcp_modules
    return mcp_modules()


def collect_tools(sess: Session) -> dict[str, RestTool]:
    reg = _Collector()
    for mod in tool_modules():
        mod.register(reg, sess)
    return reg.tools
