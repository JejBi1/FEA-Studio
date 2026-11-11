"""Tests of the reproducibility of the nonlinear calculations (B6)."""
from __future__ import annotations

import os
from pathlib import Path
import numpy as np

from fea_core import config
from fea_core.solve import run_study
import verification.cases as C


def test_ccx_env_deterministic():
    """Check that config.ccx_env applies the deterministic switch correctly."""
    # 1. The default state without an environment variable
    old_env = os.environ.pop("FEA_CCX_DETERMINISTIC", None)
    try:
        env_def = config.ccx_env(deterministic=False)
        assert "MKL_CBWR" not in env_def
        assert "KMP_DETERMINISTIC_REDUCTION" not in env_def

        # 2. Explicit deterministic=True
        env_det = config.ccx_env(deterministic=True)
        assert env_det["MKL_CBWR"] == "AUTO,STRICT"
        assert env_det["OMP_DYNAMIC"] == "FALSE"
        assert env_det["OMP_SCHEDULE"] == "static"
        assert env_det["KMP_DETERMINISTIC_REDUCTION"] == "yes"

        # 3. Control through the environment variable
        os.environ["FEA_CCX_DETERMINISTIC"] = "1"
        env_from_os = config.ccx_env()
        assert env_from_os["MKL_CBWR"] == "AUTO,STRICT"
        assert env_from_os["KMP_DETERMINISTIC_REDUCTION"] == "yes"

        os.environ["FEA_CCX_DETERMINISTIC"] = "0"
        env_off = config.ccx_env()
        assert "MKL_CBWR" not in env_off
    finally:
        if old_env is not None:
            os.environ["FEA_CCX_DETERMINISTIC"] = old_env
        else:
            os.environ.pop("FEA_CCX_DETERMINISTIC", None)


def test_nonlinear_identical_runs(tmp_path: Path):
    """Check that two identical runs of a small nonlinear case with the deterministic setting give the same results."""
    old_env = os.environ.get("FEA_CCX_DETERMINISTIC")
    os.environ["FEA_CCX_DETERMINISTIC"] = "1"
    try:
        wd1 = tmp_path / "run1"
        wd2 = tmp_path / "run2"
        wd1.mkdir()
        wd2.mkdir()

        case_pl1 = C.CASES["PL1"]
        st1, _ = case_pl1.build(wd1)
        st2, _ = case_pl1.build(wd2)

        res1 = run_study(st1, wd1 / "calc")
        res2 = run_study(st2, wd2 / "calc")

        assert len(res1.steps) == len(res2.steps)
        step1 = res1.steps[-1]
        step2 = res2.steps[-1]

        # The displacements and the stresses must agree numerically
        np.testing.assert_allclose(step1.U, step2.U, rtol=1e-7, atol=1e-9)
        np.testing.assert_allclose(step1.S, step2.S, rtol=1e-7, atol=1e-9)
        if "peeq" in step1.fields and "peeq" in step2.fields:
            np.testing.assert_allclose(step1.fields["peeq"], step2.fields["peeq"], rtol=1e-7, atol=1e-9)
    finally:
        if old_env is not None:
            os.environ["FEA_CCX_DETERMINISTIC"] = old_env
        else:
            os.environ.pop("FEA_CCX_DETERMINISTIC", None)
