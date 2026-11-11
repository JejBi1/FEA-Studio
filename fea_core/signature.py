"""Signature of the meshing inputs – shared by the GUI and MCP (spec F8, 4.2: earlier separately in main.MESH_KEYS and
Session.mesh_signature; the new contact keys must hold for both). A matching signature = a finished mesh can be used.

The mesh is affected by: geometry, selections, mesh settings, bolts / welds / joints, ignored parts and from the contacts only what
changes the meshing (parts and selections of the pairs, type, on/off, slave, way of meshing the interface, search tolerance). μ, K, λ,
the interference and the settings of the steps do not change the mesh.
"""
from __future__ import annotations

import hashlib
import json

BASE_KEYS = ("selections", "mesh", "bolts", "welds", "joints", "ignored_parts")
CONTACT_MESH_FIELDS = ("name", "a", "b", "kind", "enabled", "slave")


def mesh_inputs(study, geometry: bool = True) -> dict:
    d = study.to_dict()
    out = {k: d.get(k) for k in BASE_KEYS}
    if geometry:
        out["geometry"] = d.get("geometry")
    if d.get("imprints"):                         # bolt-free split of faces (only when they are – an older signature does not change)
        out["imprints"] = d["imprints"]
    if study.contacts or study.joints.contact_joint:
        out["contacts"] = [{k: c.get(k) for k in CONTACT_MESH_FIELDS} for c in d.get("contacts", [])]
        cs = d.get("contact_settings") or {}
        out["contact_mesh"] = {"split": cs.get("split"), "detect_tol": cs.get("detect_tol")}
    return out


def mesh_signature(study, geometry: bool = True) -> str:
    blob = json.dumps(mesh_inputs(study, geometry), sort_keys=True, default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()
