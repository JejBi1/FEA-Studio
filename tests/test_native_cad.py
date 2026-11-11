"""Module native: reading .ipt / .f3d without the CAD (fea_core.native_cad).

The Autodesk sample files are not in the repository (foreign models) – the tests against them run only where they lie in
scratch/native_cad/…/samples, otherwise they are skipped. The other tests (module, errors, Zstandard, cache) always run.
"""
from __future__ import annotations

import ctypes
import io
import zipfile
from pathlib import Path

import pytest

from fea_core import modules
from fea_core import native_cad
from fea_core.native_cad import NativeError, read_native, to_step

ROOT = Path(__file__).resolve().parent.parent
FUSION = ROOT / "scratch" / "native_cad" / "fusion" / "samples"
INVENTOR = ROOT / "scratch" / "native_cad" / "inv" / "samples"


@pytest.fixture
def enabled(monkeypatch, tmp_path):
    monkeypatch.setattr(modules, "_installed", lambda key: True)
    monkeypatch.setattr(native_cad, "cache_dir", lambda: tmp_path / "cache")


def _vol(shape) -> float:
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    g = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, g)
    return g.Mass()


def test_module_required(monkeypatch, tmp_path):
    monkeypatch.setattr(modules, "_installed", lambda key: key != "native")
    p = tmp_path / "x.ipt"
    p.write_bytes(b"x")
    assert not modules.available("native")
    with pytest.raises(modules.ModuleMissing, match="native"):
        to_step(p)


def test_bad_files_give_clear_errors(enabled, tmp_path):
    for name, data, match in (("a.ipt", b"neni ole" * 100, "is not an Inventor part"),
                              ("b.f3d", b"neni zip", "is not a Fusion archive"),
                              ("c.iam", b"x" * 600, "is not an Inventor assembly")):
        p = tmp_path / name
        p.write_bytes(data)
        with pytest.raises(NativeError, match=match):
            read_native(p)
    p = tmp_path / "prazdny.f3d"                         # a valid zip without B-rep
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("Manifest.dat", b"x")
    with pytest.raises(NativeError, match="contains no solid"):
        read_native(p)


def test_zstd_member_roundtrip():
    """A zip member with method 93 (Zstandard) through the library from cadquery-ocp."""
    from fea_core.native_cad import zstd
    lib = zstd._lib()
    src = b"ASM BinaryFile4" + bytes(range(256)) * 50
    lib.ZSTD_compressBound.restype = ctypes.c_size_t
    lib.ZSTD_compress.restype = ctypes.c_size_t
    lib.ZSTD_compress.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    buf = ctypes.create_string_buffer(lib.ZSTD_compressBound(len(src)))
    n = lib.ZSTD_compress(buf, len(buf), src, len(src), 3)
    assert zstd.decompress(buf.raw[:n], len(src)) == src
    with pytest.raises(RuntimeError):
        zstd.decompress(buf.raw[:n], len(src) + 1)


def test_rejects_invalid_solid(enabled, monkeypatch, tmp_path):
    """A body with a negative volume (wrong orientation of the faces) stops the import – no silently wrong geometry."""
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.gp import gp_Pnt
    from fea_core.native_cad import brep, f3d
    box = BRepPrimAPI_MakeBox(10, 10, 10).Solid()
    box2 = BRepPrimAPI_MakeBox(gp_Pnt(20, 0, 0), 5, 5, 5).Solid()
    monkeypatch.setattr(f3d, "documents", lambda p: [("Dil", {"Breps.BlobParts/a.smbh": b"x"})])
    monkeypatch.setattr(brep, "solids", lambda d: ([(1, 0, box.Reversed())], []))
    p = tmp_path / "d.f3d"
    p.write_bytes(b"x")
    with pytest.raises(NativeError, match="non-positive volume"):
        read_native(p)
    monkeypatch.setattr(brep, "solids", lambda d: ([(1, 0, box)], ["helix_spl_circ: bez NURBS cache"]))
    with pytest.raises(NativeError, match="helix"):
        read_native(p)
    monkeypatch.setattr(brep, "solids", lambda d: ([(1, 0, box), (2, 0, box2)], []))
    parts = read_native(p)
    assert [x.name for x in parts] == ["Dil_1", "Dil_2"] and abs(_vol(parts[0].shape) - 1000) < 1e-6
    s1 = to_step(p)
    s2 = to_step(p)                                      # a second time from the cache
    assert s1 == s2 and s1.with_suffix(".json").is_file()
    from fea_core.geometry import read_step
    assert len(read_step(s1)) == 2


# ===================================================================== Fusion assembly (design data)
def _meta_record(guid: str, ids: list[int]) -> bytes:
    import struct
    parent = "00000000-0000-0000-0000-000000000000"
    return (b"$\x00\x00\x00" + guid.encode() + struct.pack("<I", 36) + parent.encode() + struct.pack("<I", 0)
            + struct.pack("<I", 6) + b"Fusion" + struct.pack("<I", len(ids)) + struct.pack(f"<{len(ids)}Q", *ids))


def _obj(tnum: str, oid: int, data: bytes) -> bytes:
    import struct
    return struct.pack("<I", len(tnum)) + tnum.encode() + struct.pack("<Q", oid) + data


def _ref(i: int) -> bytes:
    import struct
    return b"\x01" + struct.pack("<Q", i)


def _txt(t: str) -> bytes:
    import struct
    return struct.pack("<I", len(t)) + t.encode("utf-16-le")


def _mat(R, t) -> bytes:
    import struct
    return struct.pack("<16d", R[0][0], R[0][1], R[0][2], t[0], R[1][0], R[1][1], R[1][2], t[1],
                       R[2][0], R[2][1], R[2][2], t[2], 0, 0, 0, 1)


def _vis(on: bool) -> bytes:
    return b"\x00\x00\x80\x3fT\x1a#\x00\x01" + (b"\x01" if on else b"\x00") + b"\x00" * 6 + b"\xf0\xbf"   # −1,0 (f64)


def test_fusion_design_graph():
    """Synthetic design: root → 2 occurrences of the same component (shift, rotation), the second with the light bulb off;
    the component has 2 bodies, the second hidden; an occurrence without a name gets "<component>:<n>".
    """
    from fea_core.native_cad import f3d_design as fd
    I3 = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    Rz = ((0, -1, 0), (1, 0, 0), (0, 0, 1))
    D0, D1, C0, C1, CT, ST, B1, B2, R0, A, B, H = range(10, 22)
    objs = [_obj("300", D0, _txt("1_10")), _obj("300", D1, _txt("1_11") + _txt("Deska")),
            _obj("301", C0, _ref(A) + _ref(B) + _ref(D0)), _obj("301", C1, _ref(D1)),
            _obj("302", CT, _ref(B1) + _ref(B2) + _ref(ST) + _ref(D1)), _obj("303", ST, _txt("BREP.abc.smbh")),
            _obj("304", B1, b"\x00" * 20 + _vis(True)), _obj("304", B2, b"\x00" * 20 + _vis(False)),
            _obj("305", R0, _ref(D0) + _txt("1_18")),
            _obj("305", A, _ref(D1) + _txt("Deska:1") + _mat(I3, (1.0, 0, 0))),
            _obj("305", B, _ref(D1) + _mat(Rz, (0, 2.0, 0))),
            _obj("306", H, _ref(A) + _ref(D1) + _ref(B) + _ref(D0))]
    bulk = b"\x00" * 8 + b"".join(objs)
    meta = (_meta_record(fd.CLS_DEF, [D0, D1]) + _meta_record(fd.CLS_COMP, [C0, C1])
            + _meta_record(fd.CLS_CONTAINER, [CT]) + _meta_record("74CA4562-59F8-4C97-8AD1-F8297C21F9AA", [ST])
            + _meta_record(fd.CLS_BODY, [B1, B2]) + _meta_record(fd.CLS_OCC, [R0, A, B])
            + _meta_record(fd.CLS_HIDDEN_PATH, [H]))
    inst = fd.instances({"d": (bulk, meta)}, "d", labels={"d": "Assembly"})
    got = [(i.name, i.body, i.visible, i.matrix and (round(i.matrix[3], 9), round(i.matrix[7], 9), i.matrix[1]))
           for i in inst]
    assert got == [("Deska:1", 0, True, (1.0, 0.0, 0.0)), ("Deska:1", 1, False, (1.0, 0.0, 0.0)),
                   ("Deska:2", 0, False, (0.0, 2.0, -1.0)), ("Deska:2", 1, False, (0.0, 2.0, -1.0))]
    # nested document: occurrence with the GUID of the role → the root of the second document at the position of the occurrence
    roles = fd.redirection_roles(b'{"references":[{"relativePath":"x.f3d","properties":[{"neutronRole":'
                                 b'{"value":"E0ED4AAD-7CAA-495E-B314-990D90CAB18B"}}]}]}', {"x.f3d": "x.f3d"})
    assert roles == {"e0ed4aad-7caa-495e-b314-990d90cab18b": "x.f3d"}
    with pytest.raises(fd.DesignError):
        fd.graph(b"\x00" * 64, _meta_record(fd.CLS_BODY, [1]))


def test_fusion_assembly_sample_matches_step(enabled):
    """Assembly .f3z (nested XREF document, rotated occurrences, hidden bodies and occurrences) = STEP export from Fusion."""
    src, ref = FUSION / "ada_neokey.f3z", FUSION / "ada_neokey.step"
    if not src.is_file() or not ref.is_file():
        pytest.skip("vzorový soubor není k dispozici")
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    from fea_core.geometry import read_step

    def vc(s):
        g = GProp_GProps()
        BRepGProp.VolumeProperties_s(s, g)
        c = g.CentreOfMass()
        return g.Mass(), (c.X(), c.Y(), c.Z())

    hidden: list[str] = []
    got = read_native(src, [], hidden)
    want = {p.name: vc(p.shape) for p in read_step(ref)}
    vis = {p.name: vc(p.shape) for p in got if p.name not in hidden}
    assert set(vis) == set(want) and len(hidden) == 12
    for n, (v, c) in want.items():
        assert abs(vis[n][0] - v) <= 1e-9 * v
        assert max(abs(a - b) for a, b in zip(vis[n][1], c)) < 1e-9
    step = to_step(src)
    import json
    meta = json.loads(step.with_suffix(".json").read_text(encoding="utf-8"))
    assert sorted(p["name"] for p in meta["parts"] if p.get("hidden")) == sorted(hidden)


# ===================================================================== sestava Inventoru (.iam)
def test_inventor_masked_matrix():
    """Matrix of the .iam: mask (lower half = 1, upper = 0, both = −1) + the remaining elements f64 by rows."""
    import struct
    from fea_core.native_cad.iam import masked_matrices
    # R = diag(1, −1, −1), t = (44.4, 35, 433.6): m5 = m10 = −1, m0 = m15 = 1, m1 m2 m4 m6 m8 m9 m12–14 = 0
    ones = (1 << 0) | (1 << 5) | (1 << 10) | (1 << 15)
    zeros = (1 << 1) | (1 << 2) | (1 << 4) | (1 << 5) | (1 << 6) | (1 << 8) | (1 << 9) | (1 << 10) | (7 << 12)
    data = b"\x00" * 5 + struct.pack("<I", zeros << 16 | ones) + struct.pack("<3d", 44.4, 35.0, 433.6) + b"\x00" * 9
    got = masked_matrices(data)
    assert got == [[1.0, 0.0, 0.0, 44.4, 0.0, -1.0, 0.0, 35.0, 0.0, 0.0, -1.0, 433.6, 0.0, 0.0, 0.0, 1.0]]


VNA = ROOT / "scratch" / "native_cad" / "inv" / "dl_vna" / "gpr20_vna_holder-main"


def test_inventor_assembly_sample(enabled, tmp_path):
    """Assembly with 21 occurrences and a subassembly: 37 bodies, positions of the occurrences, the subassembly = the world positions stored in the assembly
    (checked by iam.instances), missing parts → an understandable error.
    """
    src = VNA / "assemblies" / "vna_holder.iam"
    if not src.is_file():
        pytest.skip("vzorový soubor není k dispozici")
    from fea_core.geometry import bbox
    rep: list[str] = []
    parts = read_native(src, rep)
    names = [p.name for p in parts]
    assert len(parts) == 37 and "side_panel:2" in names and "Sharp GP2Y:1" in names
    assert not any("nesouhlasí" in w for w in rep)
    by = {p.name: p for p in parts}
    b1, b2 = bbox(by["side_panel:1"].shape), bbox(by["side_panel:2"].shape)
    # mirror pair around the axis of the assembly x = 323.996 mm (shift 32.3996 cm), the same height
    assert abs((b1[0] + b1[3]) / 2 + (b2[0] + b2[3]) / 2 - 2 * 323.996) < 1e-3 and abs(b1[2] - b2[2]) < 1e-9
    import shutil
    shutil.copy(src, tmp_path / src.name)
    with pytest.raises(NativeError, match="not next to it"):
        read_native(tmp_path / src.name)


# ===================================================================== sample files (locally only)
SAMPLES = [(FUSION / "ada_2020_extrusion.f3d", FUSION / "ada_2020_extrusion.step"),
           (FUSION / "ada_feather_32u4.f3d", FUSION / "ada_feather_32u4.step"),
           (INVENTOR / "gdh_aux_strut.ipt", ROOT / "scratch/native_cad/fusion/out/inventor_crosscheck/gdh_aux_strut.step"),
           (INVENTOR / "gdh_KFL08.ipt", ROOT / "scratch/native_cad/fusion/out/inventor_crosscheck/gdh_KFL08.step")]


@pytest.mark.parametrize("src, ref", SAMPLES, ids=[s.name for s, _ in SAMPLES])
def test_samples_match_reference_step(enabled, src, ref):
    if not src.is_file() or not ref.is_file():
        pytest.skip("vzorový soubor není k dispozici")
    from fea_core.geometry import read_step
    got, want = read_native(src), read_step(ref)
    assert len(got) == len(want)
    vg, vw = sum(_vol(p.shape) for p in got), sum(_vol(p.shape) for p in want)
    assert abs(vg - vw) <= 1e-7 * vw                     # reference STEP: rounding of the writing


def test_unsupported_sample_is_rejected(enabled):
    src = INVENTOR / "rw_pinch_valve_rotor.ipt"
    if not src.is_file():
        pytest.skip("vzorový soubor není k dispozici")
    with pytest.raises(NativeError, match="STEP"):
        read_native(src)
