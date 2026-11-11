"""Report section of the parametric study (fea_core.report.param) on a fake document."""
from types import SimpleNamespace

from fea_core.report.param import has_param, param_label, param_section
from fea_core.sweep import DesignPoint, ParametricStudyResult


class FakeDoc:
    def __init__(self):
        self.tables, self.paras = [], []

    def p(self, text="", **kw):
        self.paras.append(text)

    def table(self, header, rows, **kw):
        self.tables.append((header, rows))


def test_param_section_table():
    pts = [DesignPoint(0, {"load_factor": 1.0, "prop:S:E": 210000.0}, max_stress=50, max_disp=0.5, mass=1.2,
                       min_safety=4.7, max_stress_clean=40.0),
           DesignPoint(1, {"load_factor": 2.0, "prop:S:E": 210000.0}, converged=False, error="no convergence")]
    res = ParametricStudyResult(pts, pareto_indices=[0], keys=["load_factor", "prop:S:E"])
    sess = SimpleNamespace(param_study=res)
    assert has_param(sess) and not has_param(SimpleNamespace(param_study=None))
    D = FakeDoc()
    param_section(D, sess)
    header, rows = D.tables[0]
    assert header[:3] == ["DP", "load factor", "S: E"]
    assert "max. σ outside singularities [MPa]" in header and "f1 [Hz]" not in header     # empty KPI column left out
    assert rows[0][-1] == "*" and rows[1][3].startswith("error")
    assert len({len(r) for r in rows}) == 1 and len(rows[0]) == len(header)
    assert param_label("case_factor:Load 1") == "factor of Load 1"
