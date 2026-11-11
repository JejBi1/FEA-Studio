"""REST API of FEA Studio (F6, chapter 5.13 of the specification): `python -m fea_core.rest` – a thin layer over the tools of the MCP server
for the CAD adapters and scripts. Without PySide6 / fea_gui (the GUI is only started as a separate process).
"""
from .app import create_app

__all__ = ["create_app"]
