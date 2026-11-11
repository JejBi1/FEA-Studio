"""MCP tools of the voxel detail of 3D printing (fea_print_voxel_*) and a section of the protocol.

Fast: estimate of the size on a synthetic result (structured tet10 mesh, without Gmsh and ccx), errors of the parameters.
Slow: the bar X1 (PLA, tension 10 MPa) through MCP – calculation, detail 3×3×2 mm, images, protocol.
"""
import json

import pytest

pytest.importorskip("mcp")

from tests.test_mcp import Session, png_size, run  # noqa: E402
from tests.test_voxel import _results, _study, box_mesh  # noqa: E402

VOXEL_TOOLS = {"fea_print_voxel_estimate", "fea_print_voxel_run", "fea_print_voxel_render"}


def _run_with_session(tmp_path, body):
    """As tests.test_mcp.run, but the body also gets the session of the server (insertion of a synthetic study and results)."""
    import anyio
    from mcp import Client
    from fea_core.mcp.server import build_server

    async def main():
        srv = build_server(base_dir=tmp_path / "mcp")
        async with Client(srv, mode="legacy") as c:
            await body(Session(c), c, srv.fea_session)
    anyio.run(main)


def test_voxel_estimate_synthetic(tmp_path):
    async def body(s, c, sess):
        names = {t.name for t in (await c.list_tools()).tools}
        assert VOXEL_TOOLS <= names
        assert "No study is open" in await s.err("fea_print_voxel_estimate")
        sess.study = _study(walls=2, top_layers=3, bottom_layers=3, infill=0.2)
        sess.workdir = tmp_path / "wd"
        assert "results" in (await s.err("fea_print_voxel_estimate")).lower()
        sess.results = _results(box_mesh("A", (0, 0, 0), (12, 10, 6), 1.0))
        e = await s.ok("fea_print_voxel_estimate", size_mm=[3, 3, 2], margin_mm=0.5)
        assert e["ok"] and e["part"] == "A" and e["center"] == [6, 5, 3] and e["auto"]["safety"] == 5.0
        assert e["dof"] > e["voxels"] * 9 and sum(e["kinds"].values()) == e["voxels"] and e["memory_gb"] > 0
        assert e["spec"]["size"] == [3, 3, 2] and "fea_print_voxel_run" in e["next"]
        e4 = await s.ok("fea_print_voxel_estimate", center=[11, 5, 3], size_mm=[3, 3, 2], resolution="w/4",
                        margin_mm=0.5, step=0)
        assert e4["voxels"] > e["voxels"] and e4["center"] == [11, 5, 3]
        assert "resolution" in await s.err("fea_print_voxel_estimate", resolution="w/3")
        assert "size_mm" in await s.err("fea_print_voxel_estimate", size_mm=[1, 2])
        assert "not computed" in await s.err("fea_print_voxel_render")
    _run_with_session(tmp_path, body)


@pytest.mark.slow
def test_e2e_voxel_bar(tmp_path):
    """Bar 40×10×10 PLA (100 % infill, 2 perimeters) in tension 10 MPa: the detail 3×3×2 mm in the middle – σ in the detail ≈ global,
    safety_avg / p1 > 0, the difference of the average stress small; images and a section of the protocol.
    """
    import docx
    from verification.voxel_cases import _bar_study
    wd = tmp_path / "model"
    wd.mkdir()
    st = _bar_study(wd)
    st.save(wd / "study.json")

    async def body(s, c):
        await s.ok("fea_study_open", path=str(wd / "study.json"), workdir=str(tmp_path / "wd"))
        await s.ok("fea_solve_run", timeout_s=600)
        e = await s.ok("fea_print_voxel_estimate", center=[20, 5, 5], size_mm=[3, 3, 2], margin_mm=0.5)
        assert e["ok"], e
        r = await s.raw("fea_print_voxel_run", center=[20, 5, 5], size_mm=[3, 3, 2], margin_mm=0.5, timeout_s=900,
                        output_path=str(tmp_path / "vox.png"))
        assert not r.is_error, r.content[0].text
        d = json.loads(r.content[0].text)
        assert [x.type for x in r.content] == ["text", "image"] and png_size(tmp_path / "vox.png")[0] > 300
        sm = d["steps"][0]
        assert sm["safety_avg"] > 0 and sm["safety_p1"] > 0 and sm["min_safety_ip"] <= sm["safety_p1"] + 1e-9
        assert sm["homogenized_min_safety"] > 0 and sm["mean_stress_difference"] < 0.1, sm
        assert d["voxels"] == e["voxels"] or abs(d["voxels"] - e["voxels"]) < 0.05 * e["voxels"]
        assert any("step" in n for n in d["notes"]) and any("INDICATIVE" in n for n in d["notes"])
        for field, z in (("kind", "auto"), ("s11", 0), ("safety", None)):
            rr = await s.raw("fea_print_voxel_render", field=field, slice_z=z, lines=True,
                             output_path=str(tmp_path / f"v_{field}.png"), return_image=False)
            assert not rr.is_error, rr.content[0].text
        assert "slice_z" in await s.err("fea_print_voxel_render", slice_z=999)
        await s.ok("fea_report_word", output_path=str(tmp_path / "vox.docx"), language="en")
        await s.ok("fea_print_set", part="Tyc", settings={"walls": 3})      # a change of the study discards the detail
        assert "not computed" in await s.err("fea_print_voxel_render")
    run(tmp_path, body)
    doc = docx.Document(str(tmp_path / "vox.docx"))
    text = "\n".join(p.text for p in doc.paragraphs) + "\n".join(c.text for t in doc.tables for r in t.rows
                                                                for c in r.cells)
    assert "Print voxel detail" in text and "safety_avg" in text and "WARNING – THE VOXEL DETAIL IS INDICATIVE ONLY" in text
    assert "Voxels by type" in text and "section through critical layer" in text
