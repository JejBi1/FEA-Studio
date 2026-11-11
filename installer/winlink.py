"""Windows shortcuts (.lnk) and the known folders directly through the Windows interface (ctypes, COM IShellLinkW) – without PowerShell.

Shortcuts used to be created through `powershell -ExecutionPolicy Bypass -Command … WScript.Shell`: in companies with a group policy
(ExecutionPolicy from GPO, Constrained Language mode through AppLocker / WDAC) this failed silently, and antiviruses / EDR
report such a PowerShell call from an unknown installer as suspicious. IShellLinkW is the way in which Explorer and Inno Setup create shortcuts too. Only the standard library; it is not loaded on a system other than Windows.
"""
from __future__ import annotations

import ctypes
from ctypes import POINTER, WINFUNCTYPE, byref, c_int, c_ulong, c_void_p, c_wchar_p
from ctypes import HRESULT  # type: ignore[attr-defined]  (Windows only)

_ole32 = ctypes.OleDLL("ole32")
_shell32 = ctypes.WinDLL("shell32")

CLSID_SHELLLINK = "{00021401-0000-0000-C000-000000000046}"
IID_ISHELLLINKW = "{000214F9-0000-0000-C000-000000000046}"
IID_IPERSISTFILE = "{0000010B-0000-0000-C000-000000000046}"
FOLDERS = {"Desktop": "{B4BFCC3A-DB2C-424C-B029-7FE99A87C641}",       # also the desktop redirected to OneDrive
           "Programs": "{A77F5D77-2E2B-44C3-A6A2-ABA601054A51}",      # the user's Start menu
           "SendTo": "{8983036C-27C0-404B-8F08-102D10DCFD74}"}


class _GUID(ctypes.Structure):
    _fields_ = [("Data1", c_ulong), ("Data2", ctypes.c_ushort), ("Data3", ctypes.c_ushort), ("Data4", ctypes.c_ubyte * 8)]


def _guid(text: str) -> _GUID:
    g = _GUID()
    _ole32.CLSIDFromString(c_wchar_p(text), byref(g))
    return g


def _method(obj: c_void_p, index: int, *argtypes):
    """COM method by its position in the vtable (IUnknown 0–2); an error HRESULT → OSError."""
    vtbl = ctypes.cast(obj, POINTER(POINTER(c_void_p))).contents
    fn = WINFUNCTYPE(HRESULT, c_void_p, *argtypes)(vtbl[index])
    return lambda *a: fn(obj, *a)


def _release(obj: c_void_p) -> None:
    vtbl = ctypes.cast(obj, POINTER(POINTER(c_void_p))).contents
    WINFUNCTYPE(c_ulong, c_void_p)(vtbl[2])(obj)


def _com_init() -> None:
    try:
        _ole32.CoInitializeEx(None, 2)            # COINIT_APARTMENTTHREADED; S_FALSE = already initialised
    except OSError:
        pass                                      # RPC_E_CHANGED_MODE – the thread already has another mode, COM can be used


def known_folder(name: str) -> str:
    """Path of a known user folder (Desktop, Programs, SendTo) – the same as [Environment]::GetFolderPath."""
    p = c_wchar_p()
    hr = _shell32.SHGetKnownFolderPath(byref(_guid(FOLDERS[name])), 0, None, byref(p))
    if hr != 0:
        raise OSError(f"SHGetKnownFolderPath({name}) → 0x{hr & 0xFFFFFFFF:08X}")
    try:
        return p.value
    finally:
        _ole32.CoTaskMemFree(p)


def _shell_link() -> tuple[c_void_p, c_void_p]:
    _com_init()
    link = c_void_p()
    _ole32.CoCreateInstance(byref(_guid(CLSID_SHELLLINK)), None, 1, byref(_guid(IID_ISHELLLINKW)), byref(link))
    pf = c_void_p()
    try:
        _method(link, 0, POINTER(_GUID), POINTER(c_void_p))(byref(_guid(IID_IPERSISTFILE)), byref(pf))
    except OSError:
        _release(link)
        raise
    return link, pf


def create(path: str, target: str, args: str = "", workdir: str = "", description: str = "", icon: str = "") -> None:
    """Writes the shortcut `path` (.lnk). An error (e.g. Controlled folder access in Defender) → OSError."""
    link, pf = _shell_link()
    try:
        _method(link, 20, c_wchar_p)(target)                      # SetPath
        _method(link, 11, c_wchar_p)(args)                        # SetArguments
        if workdir:
            _method(link, 9, c_wchar_p)(workdir)                  # SetWorkingDirectory
        if description:
            _method(link, 7, c_wchar_p)(description)              # SetDescription
        if icon:
            _method(link, 17, c_wchar_p, c_int)(icon, 0)          # SetIconLocation
        _method(pf, 6, c_wchar_p, c_int)(path, 1)                 # IPersistFile::Save(fRemember)
    finally:
        _release(pf)
        _release(link)


def read(path: str) -> dict:
    """Target, arguments and working folder of an existing shortcut (for uninstalling only our own shortcuts)."""
    link, pf = _shell_link()
    try:
        _method(pf, 5, c_wchar_p, c_ulong)(path, 0)               # IPersistFile::Load(STGM_READ)
        out = {}
        for key, index in (("target", 3), ("workdir", 8), ("args", 10)):
            buf = ctypes.create_unicode_buffer(32768)
            if index == 3:                                        # GetPath(pszFile, cch, pfd, fFlags)
                _method(link, 3, c_wchar_p, c_int, c_void_p, c_ulong)(buf, len(buf), None, 0)
            else:
                _method(link, index, c_wchar_p, c_int)(buf, len(buf))
            out[key] = buf.value
        return out
    finally:
        _release(pf)
        _release(link)
