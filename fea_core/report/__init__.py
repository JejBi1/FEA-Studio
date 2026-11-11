"""Calculation report in Word (.docx, python-docx) – chapter 5.12 of the specification.

Without GUI: the pictures are drawn by the headless renderer of the MCP server (fea_core.mcp.render), the data are taken from the session
(fea_core.mcp.session.Session: study, parts, mesh, results, fatigue, assessment of the joints). The GUI and the CLI fill the session
through `session_from`.
"""
from .word import ReportOptions, build_report, session_from

__all__ = ["ReportOptions", "build_report", "session_from"]
