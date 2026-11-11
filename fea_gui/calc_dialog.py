"""Calculators without FEM (Tools menu, stage F8.2, fea_core.calc): snap-fit (Bayer), press fit (Lamé), Hertz,
friction in joints and machine parts, library of friction coefficients, mechanism benchmarks, damping from measurement and a fall as a
pulse. The form is built from the parameters of the core functions (fea_core.calc.common.kinds); the default values = verification examples
of the F8 spec (KK2, K4, K2, R7, MB1 …), so after opening one can calculate the check number at once.
"""
from __future__ import annotations

import html

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog, QFormLayout, QHBoxLayout, QLabel,
                               QLineEdit, QPushButton, QSpinBox, QStackedWidget, QTableWidgetItem, QTabWidget,
                               QVBoxLayout, QWidget)

from . import theme
from .widgets import decimal, dspin, fmt, table


class P:
    """Description of a form input: label, unit (suffix), description, decimal places, range, E-notation entry."""

    def __init__(self, label, unit="", tip="", dec=3, lo=0.0, hi=1e9, sci=False):
        self.label, self.unit, self.tip, self.dec, self.lo, self.hi, self.sci = label, unit, tip, dec, lo, hi, sci


MU_TIP = ("Friction coefficient. Choose by purpose (Friction library tab): force transmitted by friction (joint slip) → μ_min, mechanism resistance (drive, assembly, jamming) → μ_max")
# common parameters (name in the core → description); overrides for a specific calculation in PARAMS_KIND
PARAMS = {
    "F": P("Force F", " N", "Force [N]", 2, 0, 1e9),
    "mu": P("Friction μ", "", MU_TIP, 3, 0, 3),
    "E": P("Modulus E", " MPa", "Young's modulus [MPa] (secant modulus for plastics)", 0, 1, 1e7),
    "nu": P("Poisson ν", "", "Poisson's ratio", 3, 0, 0.499),
    "E1": P("E₁", " MPa", "Young's modulus of body 1 [MPa]", 0, 1, 1e7),
    "nu1": P("ν₁", "", "Poisson's ratio of body 1", 3, 0, 0.499),
    "E2": P("E₂", " MPa", "Young's modulus of body 2 [MPa]; empty = rigid body", 0, 1, 1e7),
    "nu2": P("ν₂", "", "Poisson's ratio of body 2 (empty = same as body 1)", 3, 0, 0.499),
    "R1": P("R₁", " mm", "Radius of curvature of body 1 [mm]", 3, 1e-6, 1e9),
    "R2": P("R₂", " mm", "Radius of curvature of body 2 [mm]; empty = plane, negative = concave surface", 3, -1e9, 1e9),
    "g": P("Gravitational acceleration g", " mm/s²", "Gravitational acceleration [mm/s²] (9810)", 1, 0, 1e6),
    "omega": P("Angular velocity ω", " rad/s", "Angular velocity of crank / cam [rad/s]", 3, -1e6, 1e6),
    "alpha": P("Angular acceleration ε", " rad/s²", "Angular acceleration of crank [rad/s²]", 3, -1e9, 1e9),
    "rho": P("Density ρ", " t/mm³", "Density [t/mm³] (steel 7.85e-9)", sci=True),
    "m": P("Mass m", " t", "Mass [t] (1 kg = 0.001 t)", sci=True),
    "k": P("Stiffness k", " N/mm", "Spring stiffness [N/mm]", 4, 0, 1e9),
    "orientation": P("Orientation", "", "Orientation of the horizontal plate relative to the buoyancy direction"),
    "S_mm": P("Fin spacing S", " mm", "Distance between adjacent fins [mm]", 2, 0.1, 1000.0),
    "N_fins": P("Number of fins N", "", "Total number of heat sink fins", 0, 1, 1000),
    "k_fin": P("Fin conductivity k", " W/(m·K)", "Thermal conductivity of the fin material [W/(m·K)] (aluminum ~170)", 1, 0.1, 5000.0),
    "Ts": P("Surface temperature Ts", " °C", "Body surface temperature [°C]", 1, -273.0, 1500.0),
    "Tinf": P("Ambient temperature T_inf", " °C", "Ambient air temperature [°C]", 1, -273.0, 1500.0),
    "velocity_m_s": P("Flow velocity v", " m/s", "Fluid velocity [m/s]", 2, 0.0, 500.0),
    "W_mm": P("Width W", " mm", "Plate width [mm] (empty = heat output Q not calculated)", 1, 0.0, 1e5),
    "L_mm": P("Length / height L", " mm", "Plate length / vertical wall height / pipe length [mm]", 1, 0.1, 1e5),
    "D_mm": P("Diameter D", " mm", "Cylinder or pipe diameter [mm]", 1, 0.1, 1e5),
    "H_mm": P("Fin height H", " mm", "Heat sink fin height [mm]", 1, 0.1, 1e4),
    "t_fin_mm": P("Fin thickness t", " mm", "Heat sink fin thickness [mm]", 2, 0.01, 100.0),
}
PARAMS_KIND = {
    # snap-fit
    "snap_fit.l": P("Beam length l", " mm", "Length of the hook beam from root to hook [mm]", 3, 0.01, 1e4),
    "snap_fit.h": P("Thickness h", " mm", "Beam thickness at root (in deflection direction) [mm]", 3, 0.01, 1e3),
    "snap_fit.b": P("Width b", " mm", "Beam width at root [mm]", 3, 0.01, 1e3),
    "snap_fit.E": P("Secant modulus E_s", " MPa", "Secant modulus at the given strain [MPa] (Bayer, manufacturer's stress–strain curves)", 0, 1, 1e6),
    "snap_fit.y": P("Deflection y", " mm", "Tip deflection = hook undercut [mm]", 3, 0.001, 1e3),
    "snap_fit.alpha_deg": P("Lead-in angle α", "°", "Angle of the hook lead-in face [°]", 2, 0, 89.9),
    "snap_fit.mu": P("Friction μ", "", "Friction coefficient during assembly (μ_max – mechanism resistance)", 3, 0, 3),
    "snap_fit.alpha_release_deg": P("Retention angle α'", "°", "Retention face angle [°] – separation force and permanent joint μ·tan α' ≥ 1; empty = skip"),
    "snap_fit.h_ratio": P("Thickness taper h_tip/h", "", "Thickness at tip / at root (1 = constant section, Bayer type 2 = 0.5)", 3, 0.05, 1),
    "snap_fit.b_ratio": P("Width taper b_tip/b", "", "Width at tip / at root (1 = constant section, Bayer type 3 = 0.25)", 3, 0.05, 1),
    "snap_fit.tip_rotation": P("Tip rotation correction", "", "The lead-in face rotates by θ when deflected → α + θ is used (own correction, UNVERIFIED in literature)"),
    "snap_fit.eps_allow_pct": P("Allowable strain", " %", "Allowable material strain [%] (Bayer, manufacturer data); empty = no check"),
    # press fit
    "press_fit.d": P("Joint diameter d", " mm", "Nominal joint diameter [mm]", 3, 0.01, 1e5),
    "press_fit.D": P("Hub outer diameter D", " mm", "Hub outer diameter [mm]", 3, 0.01, 1e5),
    "press_fit.L": P("Joint length L", " mm", "Press fit length [mm]", 3, 0.01, 1e5),
    "press_fit.delta": P("Interference δ (diametral)", " mm", "Diametral interference = d_shaft − d_hole [mm]", 4, 0, 10),
    "press_fit.E_hub": P("Hub E", " MPa", "Hub Young's modulus [MPa]", 0, 1, 1e7),
    "press_fit.nu_hub": P("Hub ν", "", "Hub Poisson's ratio", 3, 0, 0.499),
    "press_fit.E_shaft": P("Shaft E", " MPa", "Shaft Young's modulus [MPa]; empty = same as hub"),
    "press_fit.nu_shaft": P("Shaft ν", "", "Shaft Poisson's ratio; empty = same as hub"),
    "press_fit.d_inner": P("Shaft bore d_i", " mm", "Inner diameter of hollow shaft [mm] (0 = solid)", 3, 0, 1e5),
    "press_fit.mu": P("Friction for transmission μ", "", "Friction coefficient for torque and force transmission – μ_min (joint slip)",
                      3, 0, 3),
    "press_fit.mu_press": P("Press-in friction", "", "Friction coefficient for press-in force – μ_max; empty = μ"),
    "press_fit.alpha_hub": P("Hub expansion α", " 1/K", "Hub thermal expansion coefficient [1/K] (steel 11.5e-6) – heating temperature for assembly; empty = skip",
                             sci=True),
    "press_fit.mount_clearance": P("Assembly clearance s", " mm", "Clearance for shrink fitting [mm] (typically ≈ 0.001·d)",
                                   4, 0, 10),
    "press_fit.T_ambient": P("Ambient temperature", " °C", "Initial part temperature [°C]", 1, -273, 2000),
    "press_fit.Re_hub": P("Hub Re", " MPa", "Hub yield strength [MPa] – von Mises safety factor; empty = skip"),
    "press_fit.Re_shaft": P("Shaft Re", " MPa", "Shaft yield strength [MPa]; empty = skip"),
    # Hertz
    "sphere.F": P("Force F", " N", "Contact force [N]", 2, 0, 1e9),
    "cylinder.F_per_length": P("Force per length F'", " N/mm", "Contact force per unit contact length [N/mm]", 3, 0, 1e9),
    "cam_roller.N": P("Normal force N", " N", "Cam – roller normal force [N]", 2, 0, 1e9),
    "cam_roller.b": P("Contact width b", " mm", "Roller / cam width in contact [mm]", 3, 0.001, 1e5),
    "cam_roller.R_cam": P("Cam radius R_c", " mm", "Circular cam radius [mm]", 3, 0.001, 1e5),
    "cam_roller.r_roller": P("Roller radius r_f", " mm", "Roller radius [mm]", 3, 0.001, 1e5),
    "gear.z1": P("Pinion teeth z₁", "", "Number of pinion teeth"),
    "gear.z2": P("Gear teeth z₂", "", "Number of gear teeth"),
    "gear.m": P("Module m", " mm", "Gear module [mm]", 3, 0.05, 100),
    "gear.b": P("Width b", " mm", "Face width [mm]", 3, 0.01, 1e4),
    "gear.T1": P("Pinion torque T₁", " N·mm", "Pinion torque [N·mm]", 1, 0, 1e12),
    "gear.alpha_deg": P("Pressure angle α", "°", "Pressure angle [°] (20°)", 2, 10, 35),
    "gear.Z_eps": P("Z_ε", "", "Contact ratio factor (Hertz = 1)", 4, 0.1, 2),
    "gear.Z_beta": P("Z_β", "", "Helix angle factor (spur gears = 1)", 4, 0.1, 2),
    "gear.K": P("K = K_A·K_v·K_Hβ·K_Hα", "", "Product of ISO 6336 load factors (σ_H = σ_H0·√K)", 3, 0.1, 10),
    "gear.ha": P("Addendum h_a*", "", "Addendum coefficient (1)", 3, 0.5, 1.5),
    "impact.R": P("Sphere radius R", " mm", "Radius of the impacting sphere [mm]", 3, 0.001, 1e5),
    "impact.v0": P("Impact velocity v₀", " mm/s", "Impact velocity [mm/s] (1 m/s = 1000)", 2, 0, 1e7),
    "impact.R2": P("Wall R₂", " mm", "Wall radius of curvature [mm]; empty = plane"),
    "impact.m": P("Mass m", " t", "Sphere mass [t]; empty = solid sphere from density"),
    # friction of the parts
    "journal.r": P("Journal radius r", " mm", "Journal radius [mm]", 3, 0.001, 1e5),
    "journal.state": P("State", "", "Pressure distribution: clearance (line contact), worn-in (p ~ cos θ), close fit (uniform)"),
    "thrust_collar.r2": P("Outer radius r₂", " mm", "Outer radius of bearing face [mm]", 3, 0.001, 1e5),
    "thrust_collar.r1": P("Inner radius r₁", " mm", "Inner radius of bearing face [mm] (0 = solid pivot)",
                          3, 0, 1e5),
    "thrust_collar.state": P("State", "", "New (uniform pressure) or worn-in (uniform wear)"),
    "ball_joint.R": P("Sphere radius R", " mm", "Ball joint radius [mm]", 3, 0.001, 1e5),
    "ball_joint.case": P("Rotation axis", "", "About load axis, about perpendicular axis (p ~ cos θ), uniform pressure"),
    "pivot.a": P("Contact patch radius a", " mm", "Contact patch radius [mm] (Hertz – Hertz tab)", 4, 0.0001, 1e4),
    "pivot.distribution": P("Pressure distribution", "", "Hertzian (elliptical) or uniform"),
    "rolling_resistance.N": P("Load N", " N", "Normal load on wheel / roller [N]", 2, 0, 1e9),
    "rolling_resistance.r": P("Radius r", " mm", "Wheel / roller radius [mm]", 3, 0.001, 1e5),
    "rolling_resistance.e": P("Lever arm e", " mm", "Rolling resistance lever arm [mm]; empty = by pair"),
    "rolling_resistance.pair": P("Pair", "", "Approximate lever arm e: steel – steel 0.5 mm, polymer – steel 2 mm, hard rubber – steel 7.7 mm (Roymech)"),
    "guide_jamming.e": P("Force lever arm e", " mm", "Distance of the force application point from the guide axis [mm]", 3, 0, 1e5),
    "guide_jamming.L": P("Guide length L", " mm", "Guide length (spacing) [mm]", 3, 0.001, 1e5),
    "guide_jamming.mu": P("Friction μs", "", "Static friction coefficient in the guide (μ_max – mechanism resistance)", 3, 0, 3),
    "v_groove.beta_deg": P("Included angle β", "°", "V-groove included angle [°] (V-belt 38°)", 2, 1, 179),
    "v_groove.F": P("Force F", " N", "Force pressing into the groove [N] (optional)", 2, 0, 1e9),
    "wedge.alpha_deg": P("Wedge angle α", "°", "Wedge angle [°]", 3, 0, 89),
    "wedge.mu1": P("Friction μ₁", "", "Friction on the first wedge face (self-locking that must hold → μ_min)", 3, 0, 3),
    "wedge.mu2": P("Friction μ₂", "", "Friction on the second face; empty = μ₁"),
    "rolling_bearing.d": P("Bore diameter d", " mm", "Bearing bore diameter [mm] (needle: pitch D_m)", 2, 0.1, 1e5),
    "rolling_bearing.kind": P("Bearing type", "", "Rolling bearing type (μ per Roymech, ~0.1·C, good lubrication)"),
    "rolling_bearing.start": P("Start-up (2×)", "", "Start-up torque – up to twice"),
    "rolling_bearing.sealed": P("Seals (+0.001)", "", "Bearing with contact seals"),
    "power_screw.F": P("Axial force F", " N", "Axial force on the screw [N]", 2, 0, 1e9),
    "power_screw.d2": P("Pitch diameter d₂", " mm", "Thread pitch diameter [mm] (Tr 20×4: 18)", 3, 0.1, 1e4),
    "power_screw.P": P("Pitch P", " mm", "Thread pitch [mm]", 3, 0.01, 1e3),
    "power_screw.mu": P("Thread friction μ", "", "Thread friction coefficient (steel – steel dry 0.15–0.25, oiled 0.11–0.17; check self-locking with μ_min)", 3, 0, 3),
    "power_screw.flank_angle_deg": P("Thread included angle", "°", "Tr 30°, metric 60°, square 0°", 1, 0, 119),
    "power_screw.starts": P("Number of starts", "", "Number of thread starts (lead = starts × pitch)"),
    "power_screw.mu_c": P("Collar friction μ_c", "", "Friction at collar / bearing (0 = none)", 3, 0, 3),
    "power_screw.r_c": P("Collar radius r_c", " mm", "Mean collar radius [mm]", 3, 0, 1e4),
    "belt.wrap_deg": P("Wrap angle", "°", "Wrap angle [°]", 2, 0.1, 3600),
    "belt.F2": P("Slack-side force F₂", " N", "Known slack-side force [N]; enter F₂ OR F₁"),
    "belt.F1": P("Tight-side force F₁", " N", "Known tight-side force [N]; enter F₂ OR F₁"),
    "belt.q": P("Mass per length q", " t/mm", "Belt mass per unit length [t/mm] (centrifugal force q·v²)", sci=True),
    "belt.v": P("Belt speed v", " mm/s", "Belt speed [mm/s]", 1, 0, 1e7),
    "belt.groove_angle_deg": P("V-belt angle", "°", "V-belt groove angle [°] (38°); empty = flat belt / rope"),
    "worm.gamma_deg": P("Worm lead angle γ", "°", "Worm lead angle at pitch diameter [°]", 3, 0.01, 89),
    "worm.alpha_n_deg": P("Normal pressure angle", "°", "Normal pressure angle α_n [°]", 2, 0, 45),
    "clutch.F": P("Clamping force F", " N", "Clutch / brake actuating force [N]", 2, 0, 1e9),
    "clutch.R": P("Outer radius R", " mm", "Lining outer radius [mm]", 3, 0.01, 1e5),
    "clutch.r": P("Inner radius r", " mm", "Lining inner radius [mm]", 3, 0, 1e5),
    "clutch.mu": P("Lining friction μ", "", "Lining friction coefficient (from manufacturer's catalog – UNVERIFIED)", 3, 0, 3),
    "clutch.surfaces": P("Friction surfaces", "", "Number of friction surfaces (single-disc clutch 2)"),
    "clutch.state": P("State", "", "Worn-in (uniform wear) or new (uniform pressure)"),
    "clutch.cone_half_angle_deg": P("Cone half-angle", "°", "Cone clutch: half of the included angle [°]; empty = disc clutch"),
    "inclined_plane.alpha_deg": P("Breakaway angle α", "°", "Incline angle at which the body starts to slide [°]",
                                  2, 0, 89.9),
    # mechanismy
    "slider_crank.r": P("Crank r", " mm", "Crank length [mm]", 3, 0.001, 1e5),
    "slider_crank.l": P("Connecting rod l", " mm", "Connecting rod length [mm]", 3, 0.001, 1e5),
    "slider_crank.phi_deg": P("Crank angle φ", "°", "Crank angle from top dead center (φ = 0 → slider farthest) [°]", 3,
                              -3600, 3600),
    "slider_crank.e": P("Offset e", " mm", "Offset of slider axis from crank axis [mm]", 3, -1e5, 1e5),
    "slider_crank.F": P("Slider force F", " N", "Force on slider against +x (toward crank axis) → crank torque M = F·v/ω",
                        2, -1e9, 1e9),
    "four_bar.a": P("Crank a", " mm", "Crank length (input link) [mm]", 3, 0.001, 1e5),
    "four_bar.b": P("Coupler b", " mm", "Connecting rod length [mm]", 3, 0.001, 1e5),
    "four_bar.c": P("Rocker c", " mm", "Rocker length (output link) [mm]", 3, 0.001, 1e5),
    "four_bar.d": P("Ground link d", " mm", "Distance between fixed pivots [mm]", 3, 0.001, 1e5),
    "four_bar.theta2_deg": P("Crank angle θ₂", "°", "Crank angle from ground link [°]", 3, -3600, 3600),
    "four_bar.omega2": P("Crank angular velocity ω₂", " rad/s", "Crank angular velocity [rad/s] (0 = positions only)", 3,
                         -1e6, 1e6),
    "eccentric_cam.e": P("Eccentricity e", " mm", "Circular cam eccentricity [mm]", 3, 0.001, 1e4),
    "eccentric_cam.R_cam": P("Cam radius R_c", " mm", "Circular cam radius [mm]", 3, 0.001, 1e5),
    "eccentric_cam.r_roller": P("Roller radius r_f", " mm", "Follower roller radius [mm]", 3, 0.001, 1e5),
    "eccentric_cam.theta_deg": P("Cam rotation θ", "°", "Cam rotation (θ = 0 → maximum lift) [°]", 3,
                                 -3600, 3600),
    "eccentric_cam.F0": P("Spring preload F₀", " N", "Follower spring preload at lowest position [N]", 2, 0, 1e9),
    "eccentric_cam.b": P("Contact width b", " mm", "Roller width [mm] – Hertz pressure; empty = skip"),
    "pendulum.L": P("Pendulum length L", " mm", "Pendulum length [mm]", 3, 0.001, 1e7),
    "pendulum.theta0_deg": P("Amplitude θ₀", "°", "Initial deflection [°]", 3, 0.001, 179.9),
    "pendulum.m": P("Mass m", " t", "Mass [t] (pivot force); 0 = skip", sci=True),
    "sliding_stop.v0": P("Initial velocity v₀", " mm/s", "Initial velocity [mm/s]", 2, 0, 1e7),
    "sliding_stop.mu": P("Friction μk", "", "Kinetic friction coefficient", 3, 0, 3),
    "sliding_stop.incline_deg": P("Downhill slope α", "°", "Downward slope in the direction of motion [°] (0 = horizontal)", 2,
                                  -89, 89),
    "stick_slip.m": P("Mass m", " t", "Body mass [t] (1 kg = 0.001 t)", sci=True),
    "stick_slip.k": P("Spring stiffness k", " N/mm", "Stiffness of spring to fixed point [N/mm] (1000 N/m = 1 N/mm)", 4,
                      1e-6, 1e9),
    "stick_slip.v_belt": P("Belt speed v", " mm/s", "Belt speed [mm/s]", 3, 1e-6, 1e7),
    "stick_slip.mu_s": P("Static friction μs", "", "Static friction coefficient", 3, 0, 3),
    "stick_slip.mu_k": P("Kinetic friction μk", "", "Kinetic friction coefficient (μk ≤ μs)", 3, 0, 3),
    "rolling_cylinder.alpha_deg": P("Incline angle α", "°", "Inclined plane angle [°]", 2, 0, 89.9),
    "rolling_cylinder.R": P("Radius R", " mm", "Cylinder radius [mm] (angular acceleration only)", 3, 0.001, 1e5),
    "rolling_cylinder.shape": P("Shape", "", "Body shape (k = J/(mR²)): solid cylinder 1/2, thin tube 1, sphere 2/5, hollow sphere 2/3"),
    "rolling_cylinder.k": P("k = J/(mR²)", "", "Custom inertia ratio; empty = by shape"),
    "reduced_inertia.J_motor": P("Motor J", " t·mm²", "Motor moment of inertia [t·mm²] (1 kg·m² = 1000 t·mm²)",
                                 4, 0, 1e12),
    "reduced_inertia.J_load": P("Load J", " t·mm²", "Load moment of inertia at transmission output [t·mm²]", 4, 0,
                                1e12),
    "reduced_inertia.i": P("Gear ratio i", "", "i = n_motor / n_load", 4, 1e-6, 1e6),
    "reduced_inertia.M_load": P("Load torque M_z", " N·mm", "Load torque at output [N·mm]", 2, -1e12, 1e12),
    "reduced_inertia.eps_motor": P("Motor acceleration ε", " rad/s²", "Motor angular acceleration [rad/s²]", 3, -1e9, 1e9),
    "reduced_inertia.eta": P("Efficiency η", "", "Transmission efficiency (0; 1]", 3, 0.01, 1),
    "reduced_inertia.M_motor": P("Motor torque M_m", " N·mm", "Motor torque during start-up [N·mm]; empty = skip run-up time"),
    "reduced_inertia.n_target_rpm": P("Target speed", " 1/min", "Target motor speed [1/min] – run-up time"),
    "residual_vibration.a": P("Acceleration a", " mm/s²", "Rectangular acceleration pulse magnitude [mm/s²]", 2, 0, 1e9),
    "residual_vibration.tau": P("Pulse duration τ", " s", "Acceleration pulse duration [s]", 5, 1e-6, 1e4),
    "residual_vibration.f_n": P("Natural frequency f_n", " Hz", "Natural frequency of the vibrating part [Hz]", 3, 1e-3, 1e6),
    "s_curve.distance": P("Distance", " mm", "Travel length [mm]", 3, 1e-6, 1e9),
    "s_curve.v_max": P("Max. velocity", " mm/s", "Maximum allowable velocity [mm/s]", 2, 1e-6, 1e9),
    "s_curve.a_max": P("Max. acceleration", " mm/s²", "Maximum allowable acceleration [mm/s²]", 1, 1e-6, 1e12),
    "s_curve.j_max": P("Max. jerk", " mm/s³", "Maximum allowable jerk [mm/s³]", 0, 1e-6, 1e15),
    # damping
    "convert.zeta": P("Damping ratio ζ", "", "ζ as a fraction (0.02 = 2 %); fill in ONE quantity"),
    "convert.Q": P("Quality factor Q", "", "Q = 1/(2ζ); fill in ONE quantity"),
    "convert.eta": P("Loss factor η", "", "η = 2ζ (ccx *DAMPING, STRUCTURAL); fill in ONE quantity"),
    "convert.delta": P("Log. decrement δ", "", "δ = 2πζ/√(1 − ζ²); fill in ONE quantity"),
    "half_power.f_n": P("Resonance f_n", " Hz", "Frequency of the response peak [Hz]", 3, 1e-6, 1e9),
    "half_power.f1": P("Lower f₁ (−3 dB)", " Hz", "Frequency below the peak where the amplitude drops to 1/√2 [Hz]", 3,
                       1e-6, 1e9),
    "half_power.f2": P("Upper f₂ (−3 dB)", " Hz", "Frequency above the peak where the amplitude drops to 1/√2 [Hz]", 3,
                       1e-6, 1e9),
    "log_decrement.x0": P("Amplitude x₀", "", "Initial amplitude (any unit)", 5, 1e-12, 1e12),
    "log_decrement.xn": P("Amplitude x_n", "", "Amplitude n periods later (same unit)", 5, 1e-12, 1e12),
    "log_decrement.n": P("Number of periods n", "", "Number of periods between x₀ and x_n"),
    "rayleigh.f1": P("Frequency f₁", " Hz", "First frequency [Hz] (usually the 1st natural frequency)", 3, 1e-6, 1e9),
    "rayleigh.f2": P("Frequency f₂", " Hz", "Second frequency [Hz] (upper limit of the band of interest)", 3, 1e-6, 1e9),
    "rayleigh.zeta1": P("ζ₁ at f₁", "", "Damping ratio at f₁ (fraction, 0.02 = 2 %)", 4, 0, 1),
    "rayleigh.zeta2": P("ζ₂ at f₂", "", "Damping ratio at f₂; empty = ζ₁"),
    "rayleigh.f_check": P("Check frequencies", " Hz", "Frequencies for listing ζ(f), separated by semicolons; empty = f₁/2, √(f₁f₂), 2f₂"),
    "pcb.f_n": P("PCB natural frequency", " Hz", "First natural frequency of the board [Hz]", 2, 1e-3, 1e6),
    # fall
    "drop.h": P("Drop height h", " mm", "Free-fall height [mm]", 2, 0.001, 1e6),
    "drop.m": P("Mass m", " t", "Mass of the falling body [t] – with stiffness k it gives the pulse duration; otherwise empty"),
    "drop.k": P("Impact stiffness k", " N/mm", "Stiffness of the floor / packaging at impact [N/mm]; otherwise empty"),
    "drop.tau": P("Pulse duration τ", " s", "Half-sine pulse duration [s] (overrides m, k); empty = from m and k"),
    "drop.e": P("Restitution e", "", "Coefficient of restitution: 0 = no rebound, 1 = elastic rebound", 3, 0, 1),
}
CHOICES = {
    "horizontal_plate.orientation": [
        ("upper_hot", "Upper surface of hot plate / lower surface of cold plate (enhanced buoyancy)"),
        ("lower_hot", "Lower surface of hot plate / upper surface of cold plate (suppressed buoyancy)"),
    ],
    "journal.state": [("worn", "worn-in (p ~ cos θ)"), ("clearance", "clearance (line contact)"),
                      ("fitted", "fitted, new (uniform pressure)")],
    "thrust_collar.state": [("worn", "worn-in"), ("new", "new")],
    "clutch.state": [("worn", "worn-in (uniform wear)"), ("new", "new (uniform pressure)")],
    "ball_joint.case": [("perpendicular", "perpendicular axis (p ~ cos θ)"), ("axis", "load axis"),
                        ("uniform", "uniform pressure")],
    "pivot.distribution": [("hertz", "Hertz"), ("uniform", "uniform")],
    "rolling_resistance.pair": [("steel_steel", "steel – steel (0.5 mm)"), ("polymer_steel", "polymer – steel (2 mm)"),
                                ("rubber_steel", "hard rubber – steel (7.7 mm)")],
    "rolling_cylinder.shape": [("solid", "solid cylinder"), ("hollow", "thin tube"), ("sphere", "sphere"),
                               ("hollow_sphere", "hollow sphere")],
}
# default values = verification examples of the F8 spec (chapter 7)
EXAMPLES = {
    "vertical_plate": {"L_mm": 200, "Ts": 60, "Tinf": 20, "W_mm": 100},
    "horizontal_cylinder": {"D_mm": 50, "Ts": 80, "Tinf": 20, "L_mm": 500},
    "horizontal_plate": {"L_mm": 200, "W_mm": 100, "Ts": 60, "Tinf": 20},
    "finned_heatsink": {"H_mm": 50, "L_mm": 100, "S_mm": 8.0, "t_fin_mm": 2.0, "N_fins": 10, "Ts": 60, "Tinf": 20, "k_fin": 170},
    "forced_flat_plate": {"L_mm": 200, "velocity_m_s": 5, "Ts": 60, "Tinf": 20, "W_mm": 100},
    "forced_internal_pipe": {"D_mm": 25, "L_mm": 1000, "velocity_m_s": 2, "Ts": 80, "Tinf": 20},
    "snap_fit": {"l": 15, "h": 1.5, "b": 5, "E": 2000, "y": 0.5},
    "press_fit": {"d": 20, "D": 40, "L": 20, "delta": 0.02, "alpha_hub": 11.5e-6, "mount_clearance": 0.02},
    "sphere": {"F": 1000, "R1": 10},
    "cylinder": {"F_per_length": 100, "R1": 10},
    "cam_roller": {"N": 100, "b": 10, "R_cam": 30, "r_roller": 10},
    "gear": {"z1": 20, "z2": 40, "m": 3, "b": 10, "T1": 20000},
    "impact": {"R": 10, "v0": 1000},
    "journal": {"F": 1000, "r": 5, "mu": 0.2},
    "thrust_collar": {"F": 1000, "r2": 20, "r1": 10, "mu": 0.15},
    "ball_joint": {"F": 1000, "R": 10, "mu": 0.15},
    "pivot": {"F": 1000, "a": 0.4, "mu": 0.15},
    "rolling_resistance": {"N": 1000, "r": 50},
    "guide_jamming": {"e": 100, "L": 80, "mu": 0.3},
    "v_groove": {"mu": 0.3, "beta_deg": 38},
    "wedge": {"alpha_deg": 10, "mu1": 0.1},
    "rolling_bearing": {"F": 1000, "d": 20},
    "power_screw": {"F": 1000, "d2": 18, "P": 4, "mu": 0.1},
    "belt": {"mu": 0.2, "wrap_deg": 180, "F2": 100},
    "worm": {"gamma_deg": 5, "mu": 0.05},
    "clutch": {"F": 1000, "R": 50, "r": 30, "mu": 0.3},
    "inclined_plane": {"alpha_deg": 16.7},
    "slider_crank": {"r": 20, "l": 80, "phi_deg": 60, "omega": 100},
    "four_bar": {"a": 40, "b": 120, "c": 80, "d": 100, "theta2_deg": 60},
    "eccentric_cam": {"e": 5, "R_cam": 30, "r_roller": 10, "theta_deg": 90},
    "pendulum": {"L": 500, "theta0_deg": 60, "m": 0.001},
    "sliding_stop": {"v0": 1000, "mu": 0.2},
    "stick_slip": {"m": 0.001, "k": 1, "v_belt": 100, "mu_s": 0.4, "mu_k": 0.3},
    "rolling_cylinder": {"alpha_deg": 30, "mu": 0.3},
    "reduced_inertia": {"J_motor": 100, "J_load": 10000, "i": 10, "M_load": 1000, "eps_motor": 50},
    "residual_vibration": {"a": 1000, "tau": 0.01, "f_n": 30},
    "s_curve": {"distance": 100, "v_max": 200, "a_max": 2000, "j_max": 50000},
    "convert": {"zeta": 0.05},
    "half_power": {"f_n": 100, "f1": 98, "f2": 102},
    "log_decrement": {"x0": 1.0, "xn": 0.5, "n": 4},
    "rayleigh": {"f1": 10, "f2": 100, "zeta1": 0.02},
    "pcb": {"f_n": 100},
    "drop": {"h": 1000, "m": 0.001, "k": 100},
}

UNITS = [("_sqrtMPa", "√MPa"), ("_N_mm", "N/mm"), ("_Nmm", "N·mm"), ("_Nm", "N·m"), ("_MPa", "MPa"),
         ("_mm_s2", "mm/s²"), ("_mm_s", "mm/s"), ("_rad_s2", "rad/s²"), ("_rad_s", "rad/s"), ("_tmm2", "t·mm²"),
         ("_kgm2", "kg·m²"), ("_mm2", "mm²"), ("_mm4", "mm⁴"), ("_m_s", "m/s"), ("_1_s", "1/s"), ("_mm", "mm"),
         ("_um", "µm"), ("_us", "µs"), ("_ms", "ms"), ("_deg", "°"), ("_pct", "%"), ("_Hz", "Hz"), ("_K", "K"),
         ("_C", "°C"), ("_N", "N"), ("_s", "s"), ("_t", "t"), ("_g", "g")]
LABELS = {    # the whole key takes precedence over the stem without a unit
    "a_mm": "Contact patch radius a", "a_mm_s2": "Acceleration a", "delta_mm": "Approach δ", "delta_um": "Approach δ",
    "T_Nmm": "Transmitted torque T", "T_Nm": "Transmitted torque T", "T_s": "Time / period T",
    "s_mm": "Displacement / stroke s", "t_s": "Time t", "v_mm_s": "Velocity v", "eps_rad_s2": "Angular acceleration ε",
    "M_Nmm": "Moment M", "M_Nm": "Moment M", "F_N": "Resisting force F", "N_N": "Normal force N",
    "eps": "Strain ε", "eps_pct": "Strain ε", "eps_root_pct": "Strain at root",
    "eps_max_at_x_mm": "Location of maximum strain x", "P_N": "Deflection force P", "W_N": "Mating force W (Bayer)",
    "theta_tip_deg": "Tip rotation θ", "W_tip_rotation_N": "Mating force W with tip rotation",
    "W_design_N": "Mating force – design", "C_y": "Deflection coefficient C_y", "I_root_mm4": "Moment of inertia at root",
    "W_release_N": "Separation force W'", "W_release_tip_rotation_N": "Separation force W' with tip rotation",
    "permanent": "Permanent (μ·tan α' ≥ 1)", "permanent_tip_rotation": "Permanent with tip rotation",
    "alpha_release_eff_deg": "Effective retention angle", "eps_allow_pct": "Allowable strain",
    "utilization": "Utilization", "p_MPa": "Interface pressure p", "sigma_t_hub_bore_MPa": "Hub σ_t at bore",
    "sigma_r_hub_bore_MPa": "Hub σ_r at bore", "sigma_t_hub_outer_MPa": "Hub σ_t at outer surface",
    "sigma_t_shaft_surface_MPa": "Shaft σ_t at surface", "sigma_r_shaft_surface_MPa": "Shaft σ_r at surface",
    "sigma_t_shaft_bore_MPa": "Shaft σ_t at bore", "vm_hub_bore_MPa": "Hub σ_HMH at bore",
    "vm_shaft_MPa": "Shaft σ_HMH", "u_r_hub_mm": "Hub radial displacement", "u_r_shaft_mm": "Shaft radial displacement",
    "u_r_hub_um": "Hub radial displacement", "u_r_shaft_um": "Shaft radial displacement",
    "dD_hub_outer_mm": "Hub outer diameter increase", "F_axial_N": "Axial force (slip)",
    "F_press_N": "Press-in force", "contact_area_mm2": "Joint area", "dT_heating_K": "Hub heating ΔT",
    "T_heating_C": "Hub heating temperature", "safety_hub": "Hub safety factor against yielding",
    "safety_shaft": "Shaft safety factor against yielding", "E_star_MPa": "Combined modulus E*",
    "R_eff_mm": "Effective radius R", "p0_MPa": "Maximum pressure p₀", "p_mean_MPa": "Mean pressure",
    "tau_max_MPa": "Maximum shear stress τ_max", "tau_max_ratio": "τ_max / p₀", "z_tau_max_mm": "Depth of τ_max",
    "z_tau_max_ratio": "Depth of τ_max / a", "k_contact_N_mm": "Contact stiffness dF/dδ",
    "tau_max_body2_MPa": "τ_max in body 2", "z_tau_max_body2_mm": "Depth of τ_max in body 2",
    "b_mm": "Contact half-width b", "width_mm": "Contact width 2b", "rho_mm": "Effective radius ρ",
    "d1_mm": "Pitch diameter d₁", "d2_mm": "Pitch diameter d₂", "u": "Gear ratio u",
    "F_t_N": "Tangential force F_t", "F_n_N": "Normal force F_n", "rho1_mm": "Flank radius of curvature ρ₁",
    "rho2_mm": "Flank radius of curvature ρ₂", "rho_red_mm": "Effective radius ρ_red", "b_hertz_mm": "Contact half-width b",
    "Z_E_sqrtMPa": "Elasticity factor Z_E", "Z_H": "Zone factor Z_H", "sigma_H0_MPa": "σ_H0 (ISO 6336-2)",
    "sigma_H_MPa": "σ_H = σ_H0·√K", "eps_alpha": "Contact ratio ε_α", "p_b_mm": "Base pitch p_b",
    "single_pair_at_pitch": "Pitch point in single-tooth-pair contact", "m_t": "Mass m",
    "delta_max_mm": "Maximum approach δ_max", "delta_max_um": "Maximum approach δ_max",
    "F_max_N": "Maximum force F_max", "t_c_s": "Impact duration t_c", "t_c_us": "Impact duration t_c",
    "a_max_mm": "Maximum contact patch radius", "p0_max_MPa": "Maximum pressure p₀", "wave_transit_us": "Wave transit 2R/c",
    "mu_eff": "Effective coefficient μ'", "k": "Coefficient k", "M_range_Nmm": "Range of M (clearance … fitted)",
    "r_eff_mm": "Effective radius", "e_mm": "Rolling resistance lever arm e", "mu_equiv": "Equivalent coefficient e/r",
    "e_over_L": "Ratio e/L", "limit_e_over_L": "Jamming limit 1/(2μ)", "jams": "Jams",
    "L_min_mm": "Minimum guide length 2μe", "reserve": "Margin against jamming", "force_factor": "Required force multiplier",
    "factor": "Multiplier μ'/μ", "F_friction_N": "Friction force", "rho_sum_deg": "ρ₁ + ρ₂", "self_locking": "Self-locking",
    "margin_deg": "Angle margin", "mu": "Friction coefficient μ", "type": "Bearing type", "lead_mm": "Lead",
    "lambda_deg": "Lead angle λ", "rho_prime_deg": "Friction angle ρ'", "eta": "Efficiency / loss factor η",
    "eta_back": "Back-driving efficiency η'", "T_raise_Nmm": "Raising torque", "T_raise_Nm": "Raising torque",
    "T_lower_Nmm": "Lowering torque", "T_lower_Nm": "Lowering torque", "T_collar_Nmm": "Collar torque",
    "ratio": "Ratio", "F_centrifugal_N": "Centrifugal force q·v²", "F1_N": "Tight-side force F₁",
    "F2_N": "Slack-side force F₂", "F_transmitted_N": "Transmitted force F₁ − F₂", "r_m_mm": "Mean radius r_m",
    "p_max_MPa": "Maximum pressure", "mu_s": "Static coefficient μs", "x_mm": "Slider position x", "dx_dphi_mm": "dx/dφ",
    "beta_rod_deg": "Connecting rod angle β", "omega_rod_rad_s": "Connecting rod angular velocity",
    "a_approx_mm_s2": "Acceleration – approximate formula", "stroke_mm": "Slider stroke",
    "theta4_open_deg": "θ₄ – open branch", "theta4_crossed_deg": "θ₄ – crossed branch",
    "theta3_open_deg": "θ₃ – open branch", "theta3_crossed_deg": "θ₃ – crossed branch",
    "transmission_angle_open_deg": "Transmission angle (open)", "grashof": "Grashof condition",
    "grashof_type": "Four-bar linkage type", "lift_mm": "Total lift 2e", "pressure_angle_deg": "Pressure angle φ",
    "pressure_angle_max_deg": "Maximum pressure angle", "T_small_s": "Small-oscillation period",
    "v_bottom_mm_s": "Velocity at bottom", "N_bottom_over_mg": "Pivot force at bottom / mg",
    "N_bottom_N": "Pivot force at bottom", "decel_mm_s2": "Deceleration", "stops": "Stops", "stays": "Stays at rest",
    "x_s_mm": "Breakaway position x_s", "x_k_mm": "Slip center x_k", "x_min_mm": "Minimum position",
    "jump_mm": "Jump 2(x_s − x_k)", "amplitude_mm": "Amplitude", "t_stick_s": "Stick time", "t_slip_s": "Slip time",
    "omega_n_rad_s": "Natural angular frequency ω_n", "v_max_mm_s": "Maximum velocity", "rolls": "Rolls without slipping",
    "mu_required": "μ required for rolling", "F_friction_over_mg": "Friction force / mg",
    "J_red_tmm2": "Reflected moment of inertia", "J_red_kgm2": "Reflected moment of inertia",
    "M_load_red_Nmm": "Reflected load torque", "M_required_Nmm": "Required motor torque",
    "inertia_ratio": "Load/motor inertia ratio", "t_start_s": "Run-up time",
    "eps_start_rad_s2": "Angular acceleration during run-up", "X_mm": "Residual amplitude X",
    "X_static_mm": "Static deflection a/ω²", "tau_zero_s": "Durations τ with zero residual vibration", "t_j_s": "Acceleration ramp-up t_j",
    "t_a_s": "Constant acceleration t_a", "t_v_s": "Constant velocity t_v", "v_peak_mm_s": "Velocity reached",
    "a_peak_mm_s2": "Acceleration reached", "d_acc_mm": "Acceleration distance", "segments_s": "Segment durations",
    "reaches_v_max": "Reaches v_max", "zeta": "Damping ratio ζ", "zeta_pct": "Damping ratio ζ",
    "Q": "Quality factor Q", "delta": "Logarithmic decrement δ", "alpha_1_s": "Rayleigh α", "beta_s": "Rayleigh β",
    "v_m_s": "Impact velocity", "tau_s": "Pulse duration τ", "tau_ms": "Pulse duration τ", "A_mm_s2": "Peak acceleration A",
    "A_g": "Peak acceleration A", "dv_mm_s": "Velocity change Δv", "lift": "Lift",
}
BRANCH = {"open": "open", "crossed": "crossed", "branches": "branches", "zeta_at": "ζ at"}
SKIP_KEYS = {"formula", "source", "warnings", "units"}


NO_UNIT = {"mu_s"}                      # keys whose end only looks like a unit


def unit_of(key: str) -> tuple[str, str]:
    if key in NO_UNIT:
        return key, ""
    for suf, u in UNITS:
        if key.endswith(suf) and len(key) > len(suf):
            return key[: -len(suf)], u
    return key, ""


def label_of(key: str) -> str:
    stem, u = unit_of(key)
    lab = LABELS.get(key) or LABELS.get(stem) or stem
    return f"{lab} [{u}]" if u else lab


def value_text(v) -> str:
    if isinstance(v, bool):
        return "yes" if v else "ne"
    if isinstance(v, (int, float)):
        return fmt(float(v), 5)
    if isinstance(v, (list, tuple)):
        return "; ".join(value_text(x) for x in v)
    return str(v)


def result_rows(d: dict, prefix: str = "") -> list[tuple[str, str]]:
    """Rows of the results table (nested dictionaries expanded, lists of records by rows)."""
    rows = []
    for k, v in d.items():
        if k in SKIP_KEYS:
            continue
        if isinstance(v, dict):
            sub = BRANCH.get(k, k)
            for kk, vv in v.items():
                if isinstance(vv, dict):
                    rows += result_rows(vv, f"{prefix}{BRANCH.get(kk, kk)}: ")
                else:
                    lab = f"{sub} {kk}" if k == "zeta_at" else f"{sub}: {label_of(kk)}"
                    rows.append((prefix + lab, value_text(vv)))
        elif isinstance(v, list) and v and isinstance(v[0], dict):
            for item in v:
                vals = list(item.values())
                rows.append((prefix + str(vals[0]), " | ".join(value_text(x) for x in vals[1:])))
        else:
            rows.append((prefix + label_of(k), value_text(v)))
    return rows


def note_html(d: dict) -> str:
    parts = []
    if d.get("formula"):
        parts.append(f"<b>Vzorec:</b> {html.escape(d['formula'])}")
    if d.get("source"):
        parts.append(f"<b>Zdroj:</b> {html.escape(d['source'])}")
    for w in d.get("warnings") or []:
        parts.append(f"<span style='color:#c0392b'>! {html.escape(w)}</span>")
    return "<br>".join(parts)


def _num(text: str):
    t = text.strip().replace(",", ".").replace(" ", "")
    if not t:
        return None
    return float(t)


class KindForm(QWidget):
    """Form of the calculator group: choice of the kind, inputs by the function parameters, results table, formula and source."""

    def __init__(self, group: str, intro: str = "", only: list | None = None):
        super().__init__()
        from fea_core.calc import kinds
        self.group = group
        self.table_kinds = {k: v for k, v in kinds(group).items() if only is None or k in only}
        self.inputs: dict[str, dict] = {}
        lay = QVBoxLayout(self)
        if intro:
            lb = QLabel(intro)
            lb.setWordWrap(True)
            lb.setProperty("muted", True)
            lay.addWidget(lb)
        self.kind = QComboBox()
        self.kind.setToolTip("Calculation type – the form below adapts")
        for k, (_, desc) in self.table_kinds.items():
            self.kind.addItem(desc[:1].upper() + desc[1:], k)
        if len(self.table_kinds) > 1:
            f = QFormLayout()
            f.addRow("Calculation", self.kind)
            lay.addLayout(f)
        else:
            self.kind.hide()
        self.stack = QStackedWidget()
        for k, (fn, _) in self.table_kinds.items():
            self.stack.addWidget(self._page(k, fn))
        self.kind.currentIndexChanged.connect(self._kind_changed)
        lay.addWidget(self.stack)
        h = QHBoxLayout()
        b = QPushButton("Calculate")
        theme.set_primary(b)
        b.setToolTip("Calculate from the entered values (closed-form formulas, no FEA)")
        b.clicked.connect(self.compute)
        self.btn_calc = b
        h.addWidget(b)
        b = QPushButton("Default example")
        b.setToolTip("Reset inputs to the verification example from the specification (F8 check values, ch. 7)")
        b.clicked.connect(self.reset)
        h.addWidget(b)
        b = QPushButton("Copy results")
        b.setToolTip("Copy the results table, formula and source to the clipboard (tab-separated text)")
        b.clicked.connect(self.copy)
        h.addWidget(b)
        if group == "convection":
            b_use = QPushButton("Use for convection")
            b_use.setToolTip("Transfers the calculated coefficient h to the convection boundary condition in the Thermal panel")
            b_use.clicked.connect(self.apply_to_thermal)
            h.addWidget(b_use)
        h.addStretch(1)
        lay.addLayout(h)
        self.out = table(["Quantity", "Value"], min_height=180, stretch_last=True)
        self.out.setToolTip("Calculation results (units in square brackets)")
        lay.addWidget(self.out, 1)
        self.note = QLabel()
        self.note.setWordWrap(True)
        self.note.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.note.setToolTip("Formula used, source and warnings")
        lay.addWidget(self.note)
        self.last: dict | None = None

    def _kind_changed(self, i):
        self.stack.setCurrentIndex(i)
        self.out.setRowCount(0)
        self.note.clear()

    def _spec(self, kind: str, name: str) -> P:
        return PARAMS_KIND.get(f"{kind}.{name}") or PARAMS.get(name) or P(name, "", f"Parameter {name}")

    def _page(self, kind: str, fn) -> QWidget:
        from fea_core.calc.common import parameters
        w = QWidget()
        f = QFormLayout(w)
        f.setContentsMargins(0, 0, 0, 0)
        ex = EXAMPLES.get(kind, {})
        self.inputs[kind] = {}
        for p in parameters(fn):
            sp = self._spec(kind, p.name)
            ann = str(p.annotation)
            val = ex.get(p.name, None if p.default is p.empty else p.default)
            key = f"{kind}.{p.name}"
            if key in CHOICES:
                wd = QComboBox()
                for v, t in CHOICES[key]:
                    wd.addItem(t, v)
                if val is not None:
                    wd.setCurrentIndex(max(0, wd.findData(val)))
            elif p.name == "kind" and kind == "rolling_bearing":
                from fea_core.calc.friction import BEARING_MU
                wd = QComboBox()
                for v, (mu, t) in BEARING_MU.items():
                    wd.addItem(f"{t} (μ {fmt(mu)})", v)
                wd.setCurrentIndex(max(0, wd.findData(val)))
            elif ann == "bool":
                wd = QCheckBox()
                wd.setChecked(bool(val))
            elif ann == "int":
                wd = QSpinBox()
                wd.setRange(1, 100000)
                wd.setValue(int(val if val is not None else 1))
            elif "None" in ann or "list" in ann or sp.sci or p.default is None:
                wd = QLineEdit()
                wd.setPlaceholderText("not specified")
                if val is not None:
                    wd.setText(fmt(val, 6) if not sp.sci else decimal(f"{val:g}"))
            else:
                v0 = float(val if val is not None else 0.0)
                wd = dspin(v0, min(sp.lo, v0), max(sp.hi, v0), sp.dec, 10 ** -min(sp.dec, 3) if sp.dec else 1.0,
                           sp.unit)
            tip = sp.tip + (f" [{sp.unit.strip()}]" if sp.unit and isinstance(wd, QLineEdit) else "")
            wd.setToolTip(tip)
            wd.setProperty("calc_default", val)
            lab = sp.label + (f" [{sp.unit.strip()}]" if sp.unit and isinstance(wd, QLineEdit) else "")
            f.addRow(lab, wd)
            self.inputs[kind][p.name] = (wd, p, sp)
        return w

    def current_kind(self) -> str:
        return self.kind.currentData() or next(iter(self.table_kinds))

    def values(self) -> dict:
        out = {}
        for name, (wd, p, sp) in self.inputs[self.current_kind()].items():
            if isinstance(wd, QComboBox):
                out[name] = wd.currentData()
            elif isinstance(wd, QCheckBox):
                out[name] = wd.isChecked()
            elif isinstance(wd, QSpinBox):
                out[name] = wd.value()
            elif isinstance(wd, QLineEdit):
                t = wd.text()
                if "list" in str(p.annotation):
                    items = [x for x in t.replace(",", ".").split(";") if x.strip()]
                    out[name] = [float(x) for x in items] or None
                else:
                    try:
                        out[name] = _num(t)
                    except ValueError:
                        raise ValueError(f"{sp.label}: “{t}” is not a number") from None
                if out[name] is None:
                    if p.default is p.empty:
                        raise ValueError(f"{sp.label}: value is required")
                    if "None" not in str(p.annotation):         # an empty optional field without None = default
                        out[name] = p.default
            else:
                out[name] = wd.value()
        return out

    def compute(self):
        from fea_core.calc import CalcError, call
        try:
            d = call(self.group, self.current_kind(), self.values())
        except (CalcError, ValueError) as e:
            self.last = None
            self.out.setRowCount(0)
            self.note.setText(f"<span style='color:#c0392b'>{html.escape(str(e))}</span>")
            return None
        self.show_result(d)
        return d

    def show_result(self, d: dict):
        self.last = d
        rows = result_rows(d)
        self.out.setRowCount(len(rows))
        for i, (a, b) in enumerate(rows):
            self.out.setItem(i, 0, QTableWidgetItem(a))
            it = QTableWidgetItem(b)
            it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.out.setItem(i, 1, it)
        self.note.setText(note_html(d))

    def reset(self):
        for name, (wd, p, sp) in self.inputs[self.current_kind()].items():
            val = wd.property("calc_default")
            if isinstance(wd, QComboBox):
                wd.setCurrentIndex(max(0, wd.findData(val)))
            elif isinstance(wd, QCheckBox):
                wd.setChecked(bool(val))
            elif isinstance(wd, QSpinBox):
                wd.setValue(int(val or 1))
            elif isinstance(wd, QLineEdit):
                wd.setText("" if val is None else (decimal(f"{val:g}") if sp.sci else fmt(val, 6)))
            else:
                wd.setValue(float(val or 0.0))

    def apply_to_thermal(self):
        if not self.last or "h" not in self.last:
            self.compute()
        if self.last and "h" in self.last:
            p = self.parent()
            while p and not hasattr(p, "win"):
                p = p.parent()
            win = getattr(p, "win", None)
            if win and hasattr(win, "thermal_panel"):
                tp = win.thermal_panel
                idx = tp.bc_type.findData("convection")
                if idx >= 0:
                    tp.bc_type.setCurrentIndex(idx)
                tp.bc_val.setValue(float(self.last["h"]))
                tinf = self.values().get("Tinf")
                if tinf is not None:
                    tp.bc_amb.setValue(float(tinf))
                self.note.setText(note_html(self.last) + "<br><b>The coefficient h was transferred to the Convection condition (Heat panel).</b>")

    def copy(self):
        lines = []
        for r in range(self.out.rowCount()):
            lines.append(f"{self.out.item(r, 0).text()}\t{self.out.item(r, 1).text()}")
        if self.last:
            lines += ["", self.last.get("formula", ""), self.last.get("source", "")]
            lines += self.last.get("warnings") or []
        QApplication.clipboard().setText("\n".join(lines))


class FrictionLibraryTab(QWidget):
    """Library of friction coefficients: table of pairs with sources, suggestion by materials, μ by purpose, custom pairs."""

    COLS = ["Pair", "μs", "μk", "μ_min (slip)", "μ_max (resistance)", "Note", "Source"]

    def __init__(self, win=None):
        super().__init__()
        from fea_core.calc import friction_lib as fl
        self.fl = fl
        self.win = win
        lay = QVBoxLayout(self)
        lb = QLabel("Indicative friction coefficients from sources (F8 specification, table 3.3.3). Choose by purpose: "
                    + " ".join(fl.RULES[:3]) + " Values marked * are estimates (UNVERIFIED).")
        lb.setWordWrap(True)
        lb.setProperty("muted", True)
        lay.addWidget(lb)
        f = QFormLayout()
        h = QHBoxLayout()
        self.mat_a, self.mat_b = QComboBox(), QComboBox()
        names = self._material_names()
        for cb, tip in ((self.mat_a, "First material of the pair – study material or family (steel, aluminum, PLA …)"),
                        (self.mat_b, "Second material of the pair – study material or family")):
            cb.setEditable(True)
            cb.addItems(names)
            cb.setToolTip(tip)
            h.addWidget(cb, 1)
        f.addRow("Materials", h)
        h = QHBoxLayout()
        self.lub = QCheckBox("lubricated")
        self.lub.setToolTip("Lubricated contact (oil, grease)")
        h.addWidget(self.lub)
        self.cls = QComboBox()
        self.cls.addItem("no surface class", "")
        for c in "ABCD":
            self.cls.addItem(f"surface class {c} (EN 1090-2)", c)
        self.cls.setToolTip("Friction surface class of steel slip-resistant connections (EN 1090-2 Table 18, EN 1993-1-8 Table 3.7)")
        h.addWidget(self.cls)
        self.purpose = QComboBox()
        for k, t in fl.PURPOSES.items():
            self.purpose.addItem(t, k)
        self.purpose.setToolTip("Purpose: joint slip (force transfer by friction) → μ_min; mechanism resistance → μ_max")
        h.addWidget(self.purpose, 1)
        f.addRow("Conditions", h)
        lay.addLayout(f)
        h = QHBoxLayout()
        b = QPushButton("Suggest μ")
        theme.set_primary(b)
        b.setToolTip("Find the pair in the library by material families and list μ for the purpose")
        b.clicked.connect(self.suggest)
        self.btn_suggest = b
        h.addWidget(b)
        h.addStretch(1)
        lay.addLayout(h)
        self.answer = QLabel()
        self.answer.setWordWrap(True)
        self.answer.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.answer.setToolTip("Suggested friction coefficient and its source")
        lay.addWidget(self.answer)
        self.tab = table(self.COLS, min_height=220, stretch_last=True)
        self.tab.setToolTip("Friction coefficient library (built-in + user); * = estimate UNVERIFIED, – = no source found (measure)")
        self.tab.itemSelectionChanged.connect(self._row_selected)
        lay.addWidget(self.tab, 1)
        # custom pairs
        f = QFormLayout()
        self.u_name = QLineEdit()
        self.u_name.setPlaceholderText("e.g. PLA – PLA (measured)")
        self.u_name.setToolTip("Name of the custom pair (saved to the user library friction.json)")
        f.addRow("Custom pair", self.u_name)
        h = QHBoxLayout()
        self.u_a, self.u_b = QComboBox(), QComboBox()
        for cb in (self.u_a, self.u_b):
            for k, (cz, _) in fl.FAMILIES.items():
                cb.addItem(cz, k)
            cb.setToolTip("Material family (used to offer the pair in suggestions)")
            h.addWidget(cb)
        self.u_min = dspin(0.2, 0.0, 3.0, 3, 0.01)
        self.u_min.setToolTip("μ_min – joint slip (force transfer by friction)")
        self.u_max = dspin(0.3, 0.0, 3.0, 3, 0.01)
        self.u_max.setToolTip("μ_max – mechanism resistance (drive force, assembly, jamming)")
        h.addWidget(QLabel("μ_min"))
        h.addWidget(self.u_min)
        h.addWidget(QLabel("μ_max"))
        h.addWidget(self.u_max)
        f.addRow("Families and μ", h)
        self.u_src = QLineEdit()
        self.u_src.setPlaceholderText("source / measurement (inclined plane μs = tan α)")
        self.u_src.setToolTip("Source of values or description of the measurement")
        f.addRow("Source", self.u_src)
        lay.addLayout(f)
        h = QHBoxLayout()
        b = QPushButton("Save to user library")
        b.setToolTip("Save / replace the custom pair (built-in pairs cannot be overwritten)")
        b.clicked.connect(self.save_user)
        self.btn_save = b
        h.addWidget(b)
        b = QPushButton("Delete selected custom")
        b.setToolTip("Delete the selected user pair (built-in pairs cannot be deleted)")
        b.clicked.connect(self.delete_user)
        self.btn_delete = b
        h.addWidget(b)
        h.addStretch(1)
        lay.addLayout(h)
        self.msg = QLabel()
        self.msg.setWordWrap(True)
        self.msg.setToolTip("User friction library status")
        lay.addWidget(self.msg)
        self.refresh()

    def _material_names(self) -> list[str]:
        names = []
        st = getattr(self.win, "study", None)
        if st is not None:
            names += list(getattr(st, "materials", {}) or {})
        names += [cz for cz, _ in self.fl.FAMILIES.values()]
        return list(dict.fromkeys(names))

    @staticmethod
    def _rng(v):
        if not v:
            return "–"
        return fmt(v[0]) if v[0] == v[1] else f"{fmt(v[0])}–{fmt(v[1])}"

    def refresh(self):
        self.pairs = self.fl.pairs()
        t = self.tab
        t.setRowCount(len(self.pairs))
        for i, p in enumerate(self.pairs):
            star = lambda key: "*" if key in p.unverified else ""  # noqa: E731
            cells = [p.name + ("" if p.builtin else " (custom)"), self._rng(p.mu_s), self._rng(p.mu_k),
                     ("–" if p.mu_min is None else fmt(p.mu_min)) + star("mu_min"),
                     ("–" if p.mu_max is None else fmt(p.mu_max)) + star("mu_max"), p.note, p.source]
            for c, txt in enumerate(cells):
                it = QTableWidgetItem(txt)
                if c >= 5:
                    it.setToolTip(txt)
                t.setItem(i, c, it)
        self.msg.setText(f"User library: {self.fl.user_library_path()}")

    def _row_selected(self):
        r = self.tab.currentRow()
        if 0 <= r < len(self.pairs) and not self.pairs[r].builtin:
            p = self.pairs[r]
            self.u_name.setText(p.name)
            self.u_src.setText(p.source)
            for cb, fam in ((self.u_a, p.a), (self.u_b, p.b)):
                cb.setCurrentIndex(max(0, cb.findData(fam)))
            self.u_min.setValue(p.mu_min or 0.0)
            self.u_max.setValue(p.mu_max or 0.0)

    def suggest(self):
        fl = self.fl
        a, b = self.mat_a.currentText(), self.mat_b.currentText()
        res = fl.suggest(a, b, self.lub.isChecked(), self.cls.currentData() or None)
        purpose = self.purpose.currentData()
        if not res:
            fa, fb = fl.family(a), fl.family(b)
            why = ("unrecognised material family: " + ", ".join(x for x, f in ((a, fa), (b, fb)) if f is None)) \
                if fa is None or fb is None else "pair is not in the library"
            self.answer.setText(f"<span style='color:#c0392b'>{html.escape(why)} – measure (inclined plane: μs = tan α) and save as a custom pair</span>")
            return None
        best = fl.get(res[0]["name"])
        names = [r["name"] for r in res]
        for i, p in enumerate(self.pairs):
            if p.name == best.name:
                self.tab.selectRow(i)
        try:
            mu = best.mu(purpose)
            txt = (f"<b>{html.escape(best.name)}</b>: μ = {fmt(mu)} ({html.escape(fl.PURPOSES[purpose])})"
                   + (" – <span style='color:#c0392b'>estimate UNVERIFIED</span>" if best.unverified_value(purpose) else "")
                   + f"<br>Zdroj: {html.escape(best.source)}")
        except fl.FrictionLibError as e:
            mu = None
            txt = f"<span style='color:#c0392b'>{html.escape(str(e))}</span>"
        if len(names) > 1:
            txt += "<br>Other pairs: " + html.escape(", ".join(names[1:]))
        self.answer.setText(txt)
        return mu

    def save_user(self):
        fl = self.fl
        try:
            p = fl.save_pair({"name": self.u_name.text(), "a": self.u_a.currentData(), "b": self.u_b.currentData(),
                              "mu_min": self.u_min.value(), "mu_max": self.u_max.value(),
                              "source": self.u_src.text()})
        except fl.FrictionLibError as e:
            self.msg.setText(f"<span style='color:#c0392b'>{html.escape(str(e))}</span>")
            return
        self.refresh()
        self.msg.setText(f"Saved: {html.escape(p.name)} → {html.escape(str(fl.user_library_path()))}")

    def delete_user(self):
        fl = self.fl
        r = self.tab.currentRow()
        if not 0 <= r < len(self.pairs):
            self.msg.setText("Select a custom pair row")
            return
        try:
            fl.delete_pair(self.pairs[r].name)
        except fl.FrictionLibError as e:
            self.msg.setText(f"<span style='color:#c0392b'>{html.escape(str(e))}</span>")
            return
        self.refresh()


TABS = [
    ("Snap-fit", "snapfit", "Snap-fit hook of rectangular cross-section (Bayer), also tapered; mating and separation force, strain check. Default = case KK2.", None),
    ("Press fit", "pressfit", "Shaft + hub press fit (Lamé, plane stress): pressure, stresses, displacements, transmitted torque and force, heating for assembly. Default = case K4.", None),
    ("Hertz", "hertz", "Hertz contact: sphere, cylinder, gear mesh (ISO 6336-2), cam with roller follower, sphere impact. Default = cases K2, K3, KK3, KK1, KK4.", None),
    ("Machine element friction", "friction", "Friction in joints and machine elements – torque, self-locking, efficiency (F8 specification, 3.3.2). Choose μ by purpose (Friction library).", None),
    ("Mechanisms", "mechanism", "Analytical mechanism benchmarks: slider-crank (MB1), four-bar linkage (MB2), cam (KK1), pendulum (MB3), coast-down (MB5), stick-slip (MB6), rolling (MB4), axis drive.", None),
    ("Damping from measurement", "damping", "Damping ratio from measurement (−3 dB bandwidth, logarithmic decrement), Rayleigh α, β, conversions ζ ↔ Q ↔ η. Note: ccx STRUCTURAL = η = 2ζ.", None),
    ("Drop", "pulse", "Drop as an equivalent half-sine acceleration pulse (input for shock analysis).", None),
    ("Heat transfer", "convection", "Convection heat transfer calculator (D4, Churchill & Chu, Gnielinski): coefficient h [W/(m²·K)] for a vertical plate, cylinder, heat sink or pipe.", None),
]


class CalcDialog(QDialog):
    def __init__(self, win=None):
        super().__init__(win)
        self.win = win
        self.setWindowTitle("Calculators (friction, Hertz, press fit, snap-fit)")
        self.setModal(False)
        self.resize(820, 900)
        lay = QVBoxLayout(self)
        intro = QLabel("Closed-form formulas without FEA – quick estimate and analysis check. Units mm, N, MPa, t, s (g = 9810 mm/s²). Default values are verification examples.")
        intro.setWordWrap(True)
        lay.addWidget(intro)
        self.tabs = QTabWidget()
        self.tabs.setToolTip("Calculator type")
        lay.addWidget(self.tabs, 1)
        self.forms: dict[str, KindForm] = {}
        for title, group, text, only in TABS:
            fm = KindForm(group, text, only)
            self.forms[group] = fm
            self.tabs.addTab(fm, title)
            if group == "friction":
                self.library = FrictionLibraryTab(win)
                self.tabs.addTab(self.library, "Friction library")
        b = QPushButton("Close")
        b.setToolTip("Close calculators")
        b.clicked.connect(self.close)
        h = QHBoxLayout()
        h.addStretch(1)
        h.addWidget(b)
        lay.addLayout(h)


def _signature_check():                      # for tests: every parameter of the core has a description in the form
    from fea_core.calc import kinds
    from fea_core.calc.common import parameters
    missing = []
    for _, group, _, _ in TABS:
        for k, (fn, _) in kinds(group).items():
            for p in parameters(fn):
                if f"{k}.{p.name}" not in PARAMS_KIND and p.name not in PARAMS:
                    missing.append(f"{group}.{k}.{p.name}")
    return missing

