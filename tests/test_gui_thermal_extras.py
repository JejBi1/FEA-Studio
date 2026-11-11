"""GUI mapping of the heat additions (no widgets): every thermal kind has a description and a 3D colour, the cavity form ↔
CavityRadiation, one closed / ambient setting per cavity, the interface conductance with the separate mesh."""
import pytest

pytest.importorskip("PySide6")

from fea_core.study import Study  # noqa: E402
from fea_core.thermal.model import (THERMAL_KINDS, CavityRadiation, ContactConductance, Radiation,  # noqa: E402
                                    ThermalCase, FixedTemperature, HeatGeneration)


def _sample(kind):
    k = THERMAL_KINDS[kind]
    if kind == "heat_generation":
        return HeatGeneration("P", 5.0)
    if kind == "contact_conductance":
        return ContactConductance("A", "B", conductance=2000.0)
    if kind == "cavity_radiation":
        return CavityRadiation("c1", emissivity=0.7, cavity_name="Box", closed=True)
    if kind == "convection":
        return k("s", 10.0, 20.0)
    if kind == "radiation":
        return k("s", 0.8, 20.0)
    return k("s", 1.0)


def test_every_kind_is_described_and_coloured():
    from fea_gui.thermal_panel import BC_TYPES, KIND_SHORT, OTHER_COLORS, describe_bc
    for kind in THERMAL_KINDS:
        assert kind in BC_TYPES or kind in OTHER_COLORS, kind
        assert kind in KIND_SHORT, kind
        text = describe_bc(_sample(kind))
        assert text and not text.startswith(kind.capitalize() + "("), text      # not the dataclass repr
    assert "cavity “Box”" in describe_bc(_sample("cavity_radiation"))
    assert "closed" in describe_bc(_sample("cavity_radiation"))
    legacy = Radiation("x", 0.5, 30.0, cavity=True, cavity_name="Old")
    assert "cavity “Old”" in describe_bc(legacy) and "open" in describe_bc(legacy)


def test_cavity_form_mapping_and_sync():
    from fea_gui.thermal_extras import cavity_from_values, cavity_values, cavity_names, sync_cavity
    bc = cavity_from_values("hot", 0.8, 25.0, " Gap ", True)
    assert isinstance(bc, CavityRadiation) and bc.cavity_name == "Gap" and bc.closed and bc.ambient == 25.0
    assert cavity_values(bc) == {"emissivity": 0.8, "ambient": 25.0, "name": "Gap", "closed": True}
    with pytest.raises(ValueError, match="Emissivity"):
        cavity_from_values("hot", 1.2, 20.0, "Gap", True)
    with pytest.raises(ValueError, match="cavity name"):
        cavity_from_values("hot", 0.5, 20.0, "  ", True)
    case = ThermalCase("C", [FixedTemperature("x0", 100.0),
                             CavityRadiation("cold", emissivity=0.5, cavity_name="Gap", closed=False, ambient=10.0),
                             CavityRadiation("other", emissivity=0.5, cavity_name="Else", closed=False)])
    case.loads.append(bc)
    assert sync_cavity(case, bc) == ["cold"]
    assert case.loads[1].closed and case.loads[1].ambient == 25.0 and not case.loads[2].closed
    st = Study(name="t", geometry="")
    st.thermal.cases = [case]
    assert cavity_names(st) == ["Else", "Gap"]


def test_interface_conductance_makes_one_part_separate():
    from fea_core.thermal.interfaces import ConductanceError, set_interface, check_definitions
    st = Study(name="t", geometry="")
    parts = ["A", "B", "C"]
    item, sep, replaced = set_interface(st, "A", "B", 1500.0, None, parts)
    assert sep == "B" and st.mesh.separate == ["B"] and not replaced and item.name == "A – B"
    item, sep, replaced = set_interface(st, "B", "A", 3000.0, None, parts)     # same pair → replaced, no new separate
    assert sep is None and replaced and len(st.thermal.conductances) == 1 and st.thermal.conductances[0].conductance == 3000.0
    assert check_definitions(st, parts) == ([], [])
    st.mesh.separate = []
    errs, _ = check_definitions(st, parts)
    assert any("share a mesh" in e for e in errs)
    with pytest.raises(ConductanceError, match="positive"):
        set_interface(st, "A", "C", 0.0, None, parts)
    with pytest.raises(ConductanceError, match="neither"):
        set_interface(st, "A", "Z", 10.0, None, parts)
    d = Study.from_dict(st.to_dict()).thermal.conductances[0]
    assert (d.selection, d.master, d.conductance) == ("B", "A", 3000.0)


def test_old_study_without_conductances_round_trips():
    st = Study(name="t", geometry="")
    assert "conductances" not in st.to_dict()["thermal"]
