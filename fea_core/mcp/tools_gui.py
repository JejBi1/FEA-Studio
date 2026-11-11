"""Tools for live control of the FEA Studio GUI (Dynamic IPC Bridge).

When the application is open, these tools let the LLM assistant interact
directly with the user on the screen: rotate the camera, highlight elements, switch
steps of the workflow or show messages in the status bar.
If the GUI is not running, they return information about the headless state without raising an error.
"""
from __future__ import annotations


from .common import out
from .session import Session


def register(mcp, sess: Session) -> None:

    @mcp.tool(structured_output=False)
    def fea_gui_status() -> str:
        """GUI state (window open, active step, study); headless: gui_running false."""
        return out({
            "gui_running": False,
            "message": "The FEA Studio application runs in headless mode (no window is open).",
            "study": sess.study.name if sess.study is not None else None,
        })

    @mcp.tool(structured_output=False)
    def fea_gui_camera_focus(selection: str | None = None, part: str | None = None) -> str:
        """Zooms the open window's 3D camera onto a selection (name or face id 'Díl#0') or part ('*' = fit all). No arguments = Fit All."""
        return out({
            "gui_running": False,
            "focused": False,
            "message": "The FEA Studio application is not open on the desktop – the camera command was skipped.",
            "target": selection or part or "all",
        })

    @mcp.tool(structured_output=False)
    def fea_gui_highlight(entities: list[str] | None = None) -> str:
        """Highlights faces/bodies in the open window: entities = selection names or face ids 'Díl#i'; empty list clears."""
        return out({
            "gui_running": False,
            "highlighted": entities or [],
            "message": "The FEA Studio application is not open on the desktop – the highlight was skipped.",
        })

    @mcp.tool(structured_output=False)
    def fea_gui_show_mode(mode_number: int = 1) -> str:
        """Shows mode shape (vibration/buckling) number mode_number (from 1) in the 3D window."""
        return out({
            "gui_running": False,
            "mode_shown": mode_number,
            "message": "The FEA Studio application is not open on the desktop – the mode switch was skipped.",
        })

    @mcp.tool(structured_output=False)
    def fea_gui_toast(message: str, timeout_ms: int = 4000) -> str:
        """Shows `message` in the status bar of the open window for timeout_ms."""
        return out({
            "gui_running": False,
            "toast": message,
            "message": f"The GUI is not open – the message '{message}' was not shown on the screen.",
        })

    @mcp.tool(structured_output=False)
    def fea_gui_stage_switch(stage: str) -> str:
        """Switches the workflow step in the window; stage: 'Model', 'Materiály', '3D tisk', 'Spoje', 'Kontakty', 'Pruty', 'Vazby', 'Pohyb', 'Teplo', 'Zatížení', 'Dynamika', 'Síť', 'Řešení', 'Výsledky', 'Únava'."""
        return out({
            "gui_running": False,
            "stage": stage,
            "message": f"The GUI is not open – the step '{stage}' was not switched.",
        })
