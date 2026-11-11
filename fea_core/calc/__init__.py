"""Analytical calculators without a solver (stage F8.2): snap-fit (Bayer), press fit (Lamé), Hertz, friction in joints
and machine parts, library of friction coefficients, benchmark mechanisms, damping from measurement, a fall as a pulse.

Units of the project mm, N, MPa, t, s (g = 9810 mm/s²). Each function returns a dictionary: the results with the unit in the key
("p_MPa", "T_Nmm"), "formula" (short text of the formula), "source" (the source) and possibly "warnings". Pure math/numpy –
without GUI, without OCP, without a solver.

Modules: snapfit, pressfit, hertz, friction, friction_lib, mechanisms, damping, pulses; shared `common`.
"""
from .common import CalcError, KINDS, call, kinds  # noqa: F401
