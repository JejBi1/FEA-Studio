"""Parallel steps of ccx (fea_core.ccx.parallel): splitting of the .inp and the plan of the jobs (without running ccx)."""
from fea_core.ccx.parallel import ccx_memory_gb, plan, split_steps

INP = """*NODE
1, 0, 0, 0
*ELEMENT, TYPE=C3D10, ELSET=EALL
*STEP
*STATIC
*CLOAD
1, 1, 1.
*END STEP
*STEP
*STATIC
*CLOAD, OP=NEW
1, 2, 1.
*END STEP
"""


def test_split_steps():
    model, steps = split_steps(INP)
    assert model.startswith("*NODE") and "*STEP" not in model
    assert len(steps) == 2
    assert steps[0].startswith("*STEP") and "1, 1, 1." in steps[0] and "OP=NEW" not in steps[0]
    assert steps[1].startswith("*STEP") and "OP=NEW" in steps[1]
    assert model + "".join(steps) == INP


def test_plan_respects_memory_and_cores():
    assert plan(1, 200_000, 20, 8) == (1, 8)                 # jeden krok = postaru
    assert plan(5, 210_000, 20, 8) == (5, 3)                 # verified optimum 5 × 3 threads
    n, t = plan(5, 700_000, 10, 8)                           # a large model, little memory → one job
    assert n == 1
    n, t = plan(8, 50_000, 30, 8)
    assert 2 <= n <= 5 and t >= 3
    assert plan(5, 210_000, 20, 4) == (1, 4)                 # few cores
    assert ccx_memory_gb(214_000) < ccx_memory_gb(706_000)
