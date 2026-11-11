"""CLI `solve-project` and `export` (A6) and the job queue with the real solver process."""
import numpy as np
import pytest

from fea_core.cli import main
from fea_core.geometry import Part, read_step, write_step
from fea_core.project import load_project, save_project
from fea_core.selectors import find_faces
from fea_core.study import Fixed, Force, LoadCase, MeshSettings, Study
from verification.cases import STEEL, box


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    d = tmp_path_factory.mktemp("cli")
    step = d / "m.step"
    write_step([Part("Tyc", box(0, 0, 0, 40, 10, 10))], step)
    parts = read_step(step)
    st = Study("Tyc", str(step), mesh=MeshSettings(size=5.0))
    st.add_material(STEEL, default=True)
    st.add_selection("vetknuti", find_faces(parts, x=0))
    st.add_selection("konec", find_faces(parts, x=40))
    st.constraints.append(Fixed("vetknuti"))
    st.load_cases.append(LoadCase("Tah", [Force("konec", [1000.0, 0, 0])]))
    src = d / "tyc.fea"
    save_project(src, st, step)
    return src


def test_solve_project_and_export(project, tmp_path):
    out = tmp_path / "solved.fea"
    assert main(["solve-project", str(project), "-o", str(out)]) == 0
    _, _, res = load_project(out, tmp_path / "chk")
    u = res.steps[0].U[:, 0].max()
    assert u == pytest.approx(1000.0 * 40 / (100 * 210000.0), rel=0.02)        # F·L/(E·A)
    _, _, orig = load_project(project, tmp_path / "chk2")
    assert orig is None                                                          # the source is not changed
    exp = tmp_path / "exp"
    assert main(["export", str(out), "--format", "all", "--out", str(exp)]) == 0
    names = {p.suffix for p in exp.iterdir()}
    assert {".vtu", ".pvd", ".csv"} <= names
    vtu = next(exp.glob("*.vtu")).read_text(encoding="utf-8")
    assert "NumberOfPoints=\"%d\"" % len(res.mesh.nodes) in vtu
    assert main(["export", str(project), "--format", "csv", "--out", str(tmp_path / "e2")]) == 1   # no results


@pytest.mark.slow
def test_job_queue_solves_real_project(project, tmp_path, monkeypatch):
    from fea_core.job_queue import JobQueue
    monkeypatch.setattr("fea_core.config.user_data_dir", lambda: tmp_path / "ud")
    jq = JobQueue(max_workers=1)
    job = jq.submit_project(project, output=tmp_path / "q.fea")
    jq.wait_all(timeout=600)
    assert job.status == "completed", job.error
    _, _, res = load_project(job.output, tmp_path / "w")
    assert res is not None and np.isfinite(res.steps[0].U).all()
    jq.shutdown()
