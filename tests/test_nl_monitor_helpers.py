"""Data helpers of the GUI convergence monitor (fea_gui.nl_monitor) – live reading of .cvg / .sta and the series of the chart."""
import pytest

pytest.importorskip("PySide6")
from fea_gui.nl_monitor import histories_from_folder, histories_from_results, series, summary_text  # noqa: E402

CVG = """
   STEP   INC  ATT  ITER CONT.   RESID.    CORR.     RESID.    CORR.
                         EL      FORCE     DISP      FLUX      TEMP.
                                 (%)       (%)       (%)       (%)
     1     1    1    1    12   0.1000E+02 0.1000E+01 0.0000E+00 0.0000E+00
     1     1    1    2    12   0.1000E-01 0.1000E-02 0.0000E+00 0.0000E+00
     1     2    1    1    14   0.5000E+01 0.2000E+00 0.0000E+00 0.0000E+00
     1     2    2    1    14   0.3000E+01 0.1000E+00 0.0000E+00 0.0000E+00
"""
STA = """
 SUMMARY OF JOB INFORMATION
  STEP      INC     ATT  ITRS     TOT TIME     STEP TIME         INC TIME
     1        1       1     2   0.5000E+00   0.5000E+00   0.5000E+00
"""


def test_live_folder_and_series(tmp_path):
    (tmp_path / "job_s1.cvg").write_text(CVG, encoding="utf-8")
    (tmp_path / "job_s1.sta").write_text(STA, encoding="utf-8")
    h = histories_from_folder(tmp_path)
    assert list(h) == ["job_s1"]
    rf, cd, starts = series(h["job_s1"])
    assert rf == pytest.approx([10.0, 0.01, 5.0, 3.0]) and cd[0] == pytest.approx(1.0)
    assert starts == [0, 2, 3]                      # new increment, then a cutback (attempt 2)
    txt = summary_text(h, live=True)
    assert "1 increments, 4 iterations, 1 cutbacks" in txt
    assert histories_from_folder(tmp_path / "missing") == {}


def test_from_results_and_empty_text():
    class R:
        info = {"nonlinear": {"convergence": {"j": {"iterations": [], "increments": []}}}}
    assert list(histories_from_results(R())) == ["j"]
    assert histories_from_results(None) == {}
    assert "No convergence data" in summary_text({}, live=False)
