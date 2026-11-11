"""MCP server of FEA Studio for controlling FE analysis by an LLM (chapter 15 of the specification).

Start: python -m fea_core.mcp (stdio). Headless – must not import PySide6 nor fea_gui.
Modules: server (MCPServer + registration), session (state of the session), common (JSON, selections of faces, timeout),
tools_* (tools by areas), diagnostics (summary, hotspots, fields), render/raster/scenes (PNG without Qt),
prompts (resources and expert templates).
"""
