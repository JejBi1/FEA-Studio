"""Fixes from the security review before the publication (4. 11. 2026)."""
from __future__ import annotations

import math

import pytest

from fea_core import ipc
from fea_core.mcp.common import FeaError, where_filter
from fea_core.pathcheck import is_unc

BS = "\\"


@pytest.mark.parametrize("path, unc", [
    (BS * 2 + "server" + BS + "share" + BS + "a.step", True),
    ("//server/share/a.step", True),
    (BS + "??" + BS + "UNC" + BS + "server" + BS + "s" + BS + "a.step", True),       # NT namespace
    (BS + "GLOBAL??" + BS + "UNC" + BS + "server" + BS + "a", True),
    (BS + "Device" + BS + "Mup" + BS + "server" + BS + "a", True),
    ("D:" + BS + "modely" + BS + "a.step", False),
    ("modely/a.step", False),
])
def test_is_unc_rejects_nt_namespace(path, unc):
    assert is_unc(path) is unc


@pytest.mark.parametrize("expr", ["part[0]*99999", "max(part, type)*99999", "(part or 1)*9", "-part[0:2]*3",
                                  "'a'*5", "type + type"])
def test_where_rejects_text_arithmetic(expr):
    with pytest.raises(FeaError):
        where_filter(expr)


@pytest.mark.parametrize("expr", ["area*2 + cz > 1", "abs(cz)*3 < 5", "max(area, 3)*2 > 0", "part == 'A'"])
def test_where_allows_numbers(expr):
    where_filter(expr)


def test_ipc_result_mac_detects_forgery():
    token, nonce = "k" * 64, "n" * 32
    resp = {"status": "ok", "result": {"safety": 1.234567890123, "parts": ["Deska", "Žebro"], "nan": math.nan}}
    resp["rmac"] = ipc.mac(token, b"res", nonce.encode(), ipc.result_body(resp))
    import json
    back = json.loads(json.dumps(resp, ensure_ascii=False))                 # transfer through a pipe
    assert ipc.mac(token, b"res", nonce.encode(), ipc.result_body(back)) == back["rmac"]
    back["result"]["safety"] = 9.9                                          # a forged result
    assert ipc.mac(token, b"res", nonce.encode(), ipc.result_body(back)) != back["rmac"]
