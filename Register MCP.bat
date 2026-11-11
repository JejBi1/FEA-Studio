@echo off
rem Registers the FEA Studio MCP server in the LLM clients on this computer (Claude Desktop/Code, Cursor, Windsurf,
rem Antigravity, Gemini CLI, LM Studio, VS Code, Codex). Removal: "Register MCP.bat" --remove
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
".venv\Scripts\python.exe" -m fea_core.mcp.setup %*
pause
