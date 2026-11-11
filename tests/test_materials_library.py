"""Tests of the built-in material library: integrity, orthotropy, printing and the grade converter."""
from __future__ import annotations

import numpy as np
import pytest

from fea_core import grades as G
from fea_core.materials import LIBRARY, by_name
from fea_core.ortho import compliance


def test_library_not_empty():
    assert len(LIBRARY) >= 55
    names = [m.name for m in LIBRARY]
    assert len(names) == len(set(names)), "Duplicitní názvy materiálů v knihovně"


def test_materials_physical_properties():
    for m in LIBRARY:
        assert m.name.strip()
        assert m.E > 0, f"{m.name}: E <= 0"
        assert -1.0 < m.nu < 0.5, f"{m.name}: neplatné Poissonovo číslo {m.nu}"
        assert m.rho > 0, f"{m.name}: hustota <= 0"
        if m.Re is not None:
            assert m.Re > 0, f"{m.name}: Re <= 0"
        if m.Rm is not None:
            assert m.Rm > 0, f"{m.name}: Rm <= 0"
        if m.Re is not None and m.Rm is not None:
            assert m.Re <= m.Rm, f"{m.name}: Re > Rm ({m.Re} > {m.Rm})"

        # Thermal properties
        if m.k is not None:
            assert m.k > 0, f"{m.name}: k <= 0"
        if m.alpha is not None:
            assert abs(m.alpha) < 1e-3, f"{m.name}: alpha mimo rozsah {m.alpha}"
        if m.cp is not None:
            assert m.cp > 0, f"{m.name}: cp <= 0"


def test_orthotropic_composites_compliance():
    ortho_mats = [m for m in LIBRARY if m.is_ortho]
    assert len(ortho_mats) >= 8
    for m in ortho_mats:
        S = compliance(m.ortho)
        eigvals = np.linalg.eigvalsh(S)
        assert np.all(eigvals > 0), f"{m.name}: poddajnost není pozitivně definitní, vlast čísla: {eigvals}"


def test_print_materials_transverse_isotropy():
    print_mats = [m for m in LIBRARY if m.is_print]
    assert len(print_mats) >= 13
    for m in print_mats:
        assert m.E_z > 0, f"{m.name}: E_z <= 0"
        assert m.Rm_z > 0, f"{m.name}: Rm_z <= 0"


def test_am_metals_isotropic():
    """Metals for 3D printing (L-PBF) are isotropic (the print model of plastics is not used for them) and have everything for strength and heat."""
    am = {m.name: m for m in LIBRARY if "L-PBF" in m.name}
    assert len(am) >= 15
    for m in am.values():
        assert not m.is_print and not m.is_ortho, m.name
        assert m.Re and m.Rm and m.k and m.alpha and m.cp, m.name
        assert m.Re <= m.Rm, m.name
        assert "indicative" in m.source and "isotropic" in m.source, m.name
    expected = {"Ti-6Al-4V L-PBF žíhaný": "Ti-6Al-4V", "316L L-PBF bez TZ": "1.4404",
                "17-4PH L-PBF vytvrzený H900": "1.4542", "Inconel 718 L-PBF vytvrzený": "Inconel 718",
                "Inconel 625 L-PBF žíhaný": "Inconel 625", "Maraging 1.2709 L-PBF vytvrzený": "1.2709",
                "CoCrMo L-PBF bez TZ": "CoCrMo", "Ocel 1.2344 (H13) L-PBF kalená": "1.2344",
                "CuCrZr L-PBF vytvrzený": "CuCrZr", "AlSi10Mg L-PBF bez TZ": "EN AC-43000"}
    for name, key in expected.items():
        assert G.grade_of(am[name].name).key == key, name
    # heat treatment changes the strength in the right direction
    assert am["Ti-6Al-4V L-PBF HIP"].Rm < am["Ti-6Al-4V L-PBF žíhaný"].Rm < am["Ti-6Al-4V L-PBF bez TZ"].Rm
    assert am["Maraging 1.2709 L-PBF vytvrzený"].Re > 2 * am["Maraging 1.2709 L-PBF bez TZ"].Re


def test_metals_grade_mapping():
    metal_names = [
        "Ocel S235", "Ocel S275", "Ocel S355", "Ocel S690QL", "Ocel S960QL", "Ocel Hardox 450",
        "Ocel C45 +N", "Ocel 16MnCr5", "Ocel 42CrMo4 +QT", "Ocel 34CrNiMo6 +QT", "Ocel 51CrV4",
        "Ocel 100Cr6", "Ocel 1.2379", "Ocel 1.2343",
        "Nerez 1.4301", "Nerez 1.4404", "Nerez 1.4542", "Nerez 1.4462", "Nerez 1.4016", "Nerez 1.4057",
        "Šedá litina EN-GJL-150", "Šedá litina EN-GJL-250",
        "Tvárná litina EN-GJS-400-15", "Tvárná litina EN-GJS-500-7",
        "Hliník EN AW-6061 T6", "Hliník EN AW-6082 T6", "Hliník EN AW-7075 T6",
        "Hliník EN AW-2024 T3", "Hliník EN AW-5083 H111", "Hliník EN AW-6060 T6", "Hliník EN AC-43000",
        "Titan Grade 5 (Ti-6Al-4V)", "Titan Grade 2",
        "Měď Cu-ETP", "Mosaz CuZn40", "Mosaz CuZn39Pb3", "Bronz CuSn8", "Hliníkový bronz CuAl10Fe3Mn2",
        "Hořčík AZ91D", "Inconel 718",
    ]
    for name in metal_names:
        g = G.grade_of(name)
        assert g is not None, f"Materiál {name} nebyl nalezen v převodníku jakostí"
