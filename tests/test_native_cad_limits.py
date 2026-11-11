"""Safety guards of the readers of native CAD files against planted files (security review 30. 10. 2026)."""
import io
import zipfile

import pytest

from fea_core.native_cad import brep, f3d, iam


def test_linked_list_cycle_is_error():
    seen = set()
    brep._once(seen, 5)
    with pytest.raises(brep.Unsupported):
        brep._once(seen, 5)


def test_iam_reference_outside_assembly_tree_is_ignored(tmp_path):
    asm = tmp_path / "a" / "b" / "sestava"
    asm.mkdir(parents=True)
    (asm / "Sestava.iam").write_bytes(b"x")
    secret = tmp_path / "jinde" / "Tajny.ipt"
    secret.parent.mkdir()
    secret.write_bytes(b"x")
    roots = [asm, asm.parent]
    own = r"C:\Projekt\sestava\Sestava.iam"
    ref = r"C:\Projekt\sestava\..\..\..\jinde\Tajny.ipt"        # "..": outside the tree of the assembly
    assert iam.resolve(ref, own, asm / "Sestava.iam", roots) is None
    ok = asm / "Dil.ipt"
    ok.write_bytes(b"x")
    assert iam.resolve(r"C:\Projekt\sestava\Dil.ipt", own, asm / "Sestava.iam", roots) == ok.resolve()


def test_f3d_json_member_size_checked_before_inflate():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("DesignDescription.json", b" " * (f3d.MAX_JSON + 1))
    with zipfile.ZipFile(buf) as z:
        with pytest.raises(ValueError):
            f3d._small(z, "DesignDescription.json")
        assert f3d._display_names(z) == {}


def test_f3d_total_budget():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("a.bin", b"0" * 1000)
    with zipfile.ZipFile(buf) as z:
        budget = [1500]
        f3d.read_member(z, z.getinfo("a.bin"), budget)
        with pytest.raises(f3d.F3dError):
            f3d.read_member(z, z.getinfo("a.bin"), budget)
