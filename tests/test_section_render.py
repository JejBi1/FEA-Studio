"""Section in the MCP render (scenes.result_scene(section=…)): the trimmed surface + the section face with the values of the field."""
import numpy as np
import pytest

from fea_core.results import Results, StepResult
from fea_core.study import Study
from tests.test_thermal import bar_mesh


def test_result_scene_section(tmp_path):
    from fea_core.report.word import session_from
    from fea_core.mcp.scenes import result_scene
    m = bar_mesh()
    n = len(m.nodes)
    S = np.zeros((n, 6))
    S[:, 0] = m.nodes[:, 0]                       # σx = x → von Mises = x
    res = Results(m, [StepResult("L", np.zeros((n, 3)), S)])
    sess = session_from(Study("t", "x.step"), None, res, workdir=tmp_path)
    vals = res.step(0).mises
    sc, _ = result_scene(sess, 0, "mises", vals, "von Mises", "MPa", False, False, None, 12, None, None, False, False,
                         section=([50.0, 5.0, 5.0], [1.0, 0.0, 0.0]))
    assert len(sc.T) > 0
    assert sc.V[:, 0].max() <= 50.0 + 1e-6             # the side x ≤ 50 remains (the plane shifted by 1e-9·size)
    cut = np.abs(sc.V[:, 0] - 50.0) < 1e-6
    assert cut.sum() > 3 and np.allclose(sc.values[cut], 50.0, atol=1e-5)   # the section face carries the value of the field
    assert "section through point" in sc.title[0]
    with pytest.raises(ValueError):
        result_scene(sess, 0, "mises", vals, "von Mises", "MPa", False, False, None, 12, None, None, False, False,
                     section=([-10.0, 0, 0], [1.0, 0, 0]))       # x ≤ −10 would remain = nothing
