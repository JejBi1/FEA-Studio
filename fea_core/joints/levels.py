"""Effect of an edit of a bolt, a weld or the joint settings on the calculation – shared by the GUI (Joints step, undo/redo) and MCP.

Sets of joints by their effect on the calculation (fea_core.joints): MESH = inputs of the meshing (fem.mesh_inputs, finish_mesh,
open_weld_roots), FE = FE model without the mesh (stiffness / preload of the beam – fem.bolt_models, forces in the weld in the calculation –
evaluate.joint_geometry), ASSESS = only the assessment from the results (joints/assess.py, vdi2230.assess). A field that is not in
any set is taken as an input of the meshing (carefully).
"""
from __future__ import annotations

BOLT_FE = {"strength", "pitch"}                       # E, F_M,zul → stiffness and preload of the beam, α of the temperature
BOLT_PRELOAD = {"preload_mode", "preload", "alpha_A", "mu_G", "mu_K", "roughness", "DA"}  # F_V in the preload state
BOLT_ASSESS = {"mu_T", "thread_in_shear", "shank_d", "note"}
WELD_FE = {"segments", "toe_t"}                       # the sections and the reference points of the foot are computed during the calculation
WELD_ASSESS = {"fat_sigma", "fat_tau", "sn_curve", "gamma_Mf", "toe", "fat_toe", "filler", "note"}
# settings of the joints (JointSettings): only the assessment / recognition rules; preload_case changes the FE model, the others the mesh
SETTINGS_ASSESS = {"category", "slip_mu", "gamma_M2", "gamma_M3", "shear_split", "weld_colors", "weld_color_tol"}
SETTINGS_FE = {"preload_case"}
LEVELS = ("", "assess", "fe", "mesh")                 # effect of a change: nothing < only the assessment < FE calculation < mesh


def change_level(kind: str, old: dict, new: dict, preload_case: bool) -> str:
    """Effect of an edit of a bolt / weld (kind "bolt" | "weld"; old, new = to_dict()) on the calculation:
    "" no change, "assess" only the assessment (the mesh and the FE results remain valid), "fe" a new FE calculation (the mesh remains valid),
    "mesh" a new mesh. The preload (BOLT_PRELOAD) changes the FE model only with the state "Bolt preload".
    """
    diff = {k for k in set(old) | set(new) if old.get(k) != new.get(k)}
    if not diff:
        return ""
    if not old.get("enabled", True) and not new.get("enabled", True):
        return "assess"                     # a switched-off bolt / weld is not in the FE model (the parts are meshed as bodies)
    if kind == "bolt":
        fe = BOLT_FE | (BOLT_PRELOAD if preload_case else set())
        assess = BOLT_ASSESS | (set() if preload_case else BOLT_PRELOAD)
    else:
        fe, assess = WELD_FE, WELD_ASSESS
    if diff - fe - assess:
        return "mesh"
    return "fe" if diff & fe else "assess"


def settings_level(old: dict, new: dict) -> str:
    """Effect of a change of the joint settings (JointSettings.to_dict()): "" | "assess" | "fe" | "mesh"."""
    diff = {k for k in set(old) | set(new) if old.get(k) != new.get(k)}
    if not diff:
        return ""
    if diff - SETTINGS_ASSESS - SETTINGS_FE:
        return "mesh"
    return "fe" if diff & SETTINGS_FE else "assess"
