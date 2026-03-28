"""
Earth–Moon Orbital Simulation
==============================
Simulates Earth's elliptical orbit around the Sun AND the Moon's orbit around
Earth, using Kepler's equation with real astronomical values.

Physics:
  - Kepler's equation (M = E - e·sin E) solved by Newton-Raphson
  - Earth: heliocentric ellipse, Sun at one focus
  - Moon: geocentric ellipse, position added to Earth's heliocentric position
  - Velocities computed analytically from dE/dt = n / (1 - e·cos E)
  - Moon apsidal precession: perigee rotates 360° in ~8.85 years (40.6°/yr),
    implemented by rotating the perifocal frame by ω(t) = Ω_dot · t each step

Eccentricity note:
  Real values — Earth e=0.0167, Moon e=0.0549 — are so small the orbits look
  indistinguishable from circles at any reasonable plot scale. A larger
  *visual* eccentricity (ECC_E_VIS, ECC_M_VIS) is used for both the orbit
  shapes AND the motion, so the animated trail always lies on the guide ellipse.
  Real values are shown in the info boxes.
"""

import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyArrowPatch

# ---------------------------------------------------------------------------
# Kepler solver (shared by Earth and Moon)
# ---------------------------------------------------------------------------
def solve_kepler(M, ecc, tol=1e-10):
    """Newton-Raphson iteration: find eccentric anomaly E for mean anomaly M."""
    E = M.copy()
    for _ in range(50):
        dE = (M - E + ecc * np.sin(E)) / (1 - ecc * np.cos(E))
        E += dE
        if np.max(np.abs(dE)) < tol:
            break
    return E

# ---------------------------------------------------------------------------
# Earth orbital constants
# ---------------------------------------------------------------------------
A_E       = 1.000      # semi-major axis [AU]
ECC_E     = 0.0167     # TRUE eccentricity (used in readouts)
ECC_E_VIS = 0.60       # VISUAL eccentricity (exaggerated so ellipse is visible)
T_E       = 365.25     # orbital period [days]

# Derived from visual eccentricity (drive both shape and motion)
B_E   = A_E * np.sqrt(1 - ECC_E_VIS**2)
C_E   = A_E * ECC_E_VIS          # Sun focal offset [AU]
N_E   = 2 * np.pi / T_E

# Perihelion / aphelion distances using VISUAL eccentricity (for display markers)
R_PERI = A_E * (1 - ECC_E_VIS)
R_APHE = A_E * (1 + ECC_E_VIS)

# ---------------------------------------------------------------------------
# Moon orbital constants  (geocentric)
# ---------------------------------------------------------------------------
A_M       = 384_400 / 149_597_870.7  # true semi-major axis [AU] ≈ 0.002570
ECC_M     = 0.0549                    # TRUE eccentricity
ECC_M_VIS = 0.55                      # VISUAL eccentricity (exaggerated)
T_M       = 27.322                    # sidereal period [days]

B_M_VIS   = A_M * np.sqrt(1 - ECC_M_VIS**2)
N_M       = 2 * np.pi / T_M

# Apsidal precession: the Moon's perigee completes one full rotation in ~8.85 yr
T_PREC    = 8.85 * 365.25          # precession period [days]
OMEGA_DOT = 2 * np.pi / T_PREC     # precession rate   [rad/day]  ≈ 0.001942 rad/day

# Display scale: Moon's geocentric orbit is ~0.00257 AU — invisible at AU scale.
# Multiply geocentric displacement by MOON_SCALE for plotting only.
MOON_SCALE  = 28.0
MOON_PHASE0 = np.pi / 4   # Moon's mean anomaly at t=0

# ---------------------------------------------------------------------------
# Pre-compute one full Earth orbit
# ---------------------------------------------------------------------------
N_STEPS = 2000
t_arr = np.linspace(0, T_E, N_STEPS, endpoint=False)   # [days]

# --- Earth (uses ECC_E_VIS for both shape and motion) ---
E_E_arr  = solve_kepler(N_E * t_arr, ECC_E_VIS)
ex_arr   = A_E * (np.cos(E_E_arr) - ECC_E_VIS)
ey_arr   = B_E * np.sin(E_E_arr)
dEEdt    = N_E / (1 - ECC_E_VIS * np.cos(E_E_arr))
evx_arr  = -A_E * np.sin(E_E_arr) * dEEdt
evy_arr  =  B_E * np.cos(E_E_arr) * dEEdt
ev_arr   = np.sqrt(evx_arr**2 + evy_arr**2)

# True Earth–Sun distance for the readout, re-scaled to real eccentricity:
#   r_true(t) ≈ A * (1 - e_true * cos E) — approximate via same E progression
er_arr   = A_E * (1 - ECC_E * np.cos(E_E_arr))

# --- Moon (uses ECC_M_VIS; includes apsidal precession) ---
E_M_arr  = solve_kepler(N_M * t_arr + MOON_PHASE0, ECC_M_VIS)

# Argument of perigee at each time step (precession)
omega_arr = OMEGA_DOT * t_arr      # [rad], starts at 0 and grows ~40.6° over 1 year
cos_om    = np.cos(omega_arr)
sin_om    = np.sin(omega_arr)

# Position in perifocal frame (perigee along local +x')
mx_peri  = A_M * (np.cos(E_M_arr) - ECC_M_VIS)
my_peri  = B_M_VIS * np.sin(E_M_arr)

# Rotate perifocal → inertial geocentric frame by ω(t)
mx_geo   = mx_peri * cos_om - my_peri * sin_om
my_geo   = mx_peri * sin_om + my_peri * cos_om

# True Earth–Moon distance for readout (via true eccentricity)
mr_arr   = A_M * (1 - ECC_M * np.cos(E_M_arr))     # [AU]

# Display position (scaled geocentric offset + Earth heliocentric position)
mx_disp  = ex_arr + MOON_SCALE * mx_geo
my_disp  = ey_arr + MOON_SCALE * my_geo

# Moon velocity: compute in perifocal frame, then rotate
dEMdt    = N_M / (1 - ECC_M_VIS * np.cos(E_M_arr))
mvx_peri = -A_M * np.sin(E_M_arr) * dEMdt
mvy_peri =  B_M_VIS * np.cos(E_M_arr) * dEMdt
mvx_geo  = mvx_peri * cos_om - mvy_peri * sin_om
mvy_geo  = mvx_peri * sin_om + mvy_peri * cos_om
mv_arr   = np.sqrt(mvx_geo**2 + mvy_geo**2)

# ---------------------------------------------------------------------------
# Animation parameters
# ---------------------------------------------------------------------------
FRAMES_PER_ORBIT = 365
INTERVAL_MS      = 30
EARTH_TRAIL_LEN  = 60
MOON_TRAIL_LEN   = 30

frame_idx = np.linspace(0, N_STEPS - 1, FRAMES_PER_ORBIT, dtype=int)

# Unit ellipse for the Moon orbit ring (scaled/shifted each frame)
_t_ring  = np.linspace(0, 2 * np.pi, 200)
# Parametric ellipse centred at (0,0) with semi-axes 1 and b/a:
_ring_ux = np.cos(_t_ring)
_ring_uy = np.sin(_t_ring)

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(10, 9), facecolor="#0d0d1a")
ax  = fig.add_subplot(111, aspect="equal", facecolor="#0d0d1a")

MARGIN = 0.15
ax.set_xlim(-R_APHE - MARGIN, R_PERI + MARGIN)
ax.set_ylim(-B_E - MARGIN, B_E + MARGIN)
ax.set_xlabel("x [AU]", color="#aaaacc", fontsize=9)
ax.set_ylabel("y [AU]", color="#aaaacc", fontsize=9)
ax.set_title("Earth–Moon System Orbiting the Sun", color="white", fontsize=13, pad=12)
ax.tick_params(colors="#aaaacc", labelsize=8)
for spine in ax.spines.values():
    spine.set_edgecolor("#333355")
ax.grid(color="#1a1a33", linewidth=0.5, zorder=0)

# Exaggeration notice
ax.text(0.50, 0.01,
        f"Eccentricities exaggerated for visibility  "
        f"(Earth: true {ECC_E}, shown {ECC_E_VIS}  |  "
        f"Moon: true {ECC_M}, shown {ECC_M_VIS})",
        transform=ax.transAxes, fontsize=7, color="#888899",
        ha="center", va="bottom", style="italic")

# ---------------------------------------------------------------------------
# Static elements
# ---------------------------------------------------------------------------
# Earth orbit guide (dashed ellipse, using visual eccentricity)
_th = np.linspace(0, 2 * np.pi, 500)
ax.plot(A_E * np.cos(_th) - C_E, B_E * np.sin(_th),
        "--", color="#2a2a55", linewidth=0.8, zorder=1)

# Sun at origin (one focus)
ax.add_patch(plt.Circle((0, 0), 0.055, color="#FF8C00", alpha=0.30, zorder=4))
ax.add_patch(plt.Circle((0, 0), 0.034, color="#FFD700",             zorder=5))
ax.plot(0, 0, "+", color="white", markersize=5, zorder=6)

# Empty focus (second focus of Earth ellipse)
ax.plot(-2 * C_E, 0, "x", color="#555577", markersize=6, zorder=3)

# Perihelion & aphelion markers
peri_pt = np.array([ R_PERI, 0.0])
aphe_pt = np.array([-R_APHE, 0.0])
ax.plot(*peri_pt, "o", color="#ff6644", markersize=5, zorder=7)
ax.plot(*aphe_pt, "o", color="#4488ff", markersize=5, zorder=7)
ax.annotate("Perihelion\n(Jan 3)", peri_pt, textcoords="offset points",
            xytext=(6, -20), fontsize=7.5, color="#ff6644",
            arrowprops=dict(arrowstyle="-", color="#ff6644", lw=0.7))
ax.annotate("Aphelion\n(Jul 4)", aphe_pt, textcoords="offset points",
            xytext=(-68, -20), fontsize=7.5, color="#4488ff",
            arrowprops=dict(arrowstyle="-", color="#4488ff", lw=0.7))

# ---------------------------------------------------------------------------
# Dynamic elements
# ---------------------------------------------------------------------------
earth_trail, = ax.plot([], [], "-",  color="#88aaff", lw=1.2, alpha=0.6, zorder=8)
earth_dot,   = ax.plot([], [], "o",  color="#3399ff", markersize=9,
                        markeredgecolor="white", markeredgewidth=0.6, zorder=10)
vel_arrow = FancyArrowPatch((0, 0), (0, 0),
                             arrowstyle="-|>", color="#00ffaa",
                             mutation_scale=10, linewidth=1.3, zorder=11)
ax.add_patch(vel_arrow)

# Moon orbit ring (ellipse centred on Earth, updated each frame)
moon_ring, = ax.plot([], [], "-",  color="#555570", lw=0.9, alpha=0.8, zorder=8)
em_line,   = ax.plot([], [], "-",  color="#666688", lw=0.5, alpha=0.5, zorder=9)
moon_trail, = ax.plot([], [], "-", color="#ccccaa", lw=0.9, alpha=0.55, zorder=9)
moon_dot,   = ax.plot([], [], "o", color="#ddddcc", markersize=5,
                       markeredgecolor="#999988", markeredgewidth=0.5, zorder=10)

# ---------------------------------------------------------------------------
# HUD
# ---------------------------------------------------------------------------
date_text  = ax.text(0.02, 0.97, "", transform=ax.transAxes, fontsize=9,
                     color="white",   va="top", fontfamily="monospace")
dist_text  = ax.text(0.02, 0.90, "", transform=ax.transAxes, fontsize=9,
                     color="#aaddff", va="top", fontfamily="monospace")
vel_text   = ax.text(0.02, 0.83, "", transform=ax.transAxes, fontsize=9,
                     color="#00ffaa", va="top", fontfamily="monospace")
moon_text  = ax.text(0.02, 0.76, "", transform=ax.transAxes, fontsize=9,
                     color="#ddddcc", va="top", fontfamily="monospace")
prec_text  = ax.text(0.02, 0.69, "", transform=ax.transAxes, fontsize=9,
                     color="#ff9944", va="top", fontfamily="monospace")

# Info boxes
ax.text(0.975, 0.97,
        "Earth orbit (true values)\n"
        f"  a = {A_E:.4f} AU,  e = {ECC_E:.4f}\n"
        f"  T = {T_E:.2f} d\n"
        f"  q = {A_E*(1-ECC_E):.4f} AU  (perihelion)\n"
        f"  Q = {A_E*(1+ECC_E):.4f} AU  (aphelion)",
        transform=ax.transAxes, fontsize=7.5, color="#ccccdd",
        va="top", ha="right", fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#11112a",
                  edgecolor="#333366", alpha=0.85))

ax.text(0.975, 0.76,
        "Moon orbit  (true values)\n"
        f"  a = {A_M*1e3:.4f}×10⁻³ AU\n"
        f"    = {A_M*149_597_870.7:.0f} km\n"
        f"  e = {ECC_M:.4f},  T = {T_M:.3f} d\n"
        f"  Apsidal prec.: {T_PREC/365.25:.2f} yr\n"
        f"  Display scale: ×{MOON_SCALE:.0f}",
        transform=ax.transAxes, fontsize=7.5, color="#ccccaa",
        va="top", ha="right", fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#11112a",
                  edgecolor="#333355", alpha=0.85))

ax.legend(handles=[
    mpatches.Patch(color="#FFD700", label="Sun"),
    mpatches.Patch(color="#3399ff", label="Earth"),
    mpatches.Patch(color="#88aaff", label="Earth trail"),
    mpatches.Patch(color="#ddddcc", label="Moon"),
    mpatches.Patch(color="#ccccaa", label="Moon trail"),
    mpatches.Patch(color="#00ffaa", label="Earth velocity"),
], loc="lower right", fontsize=7.5, facecolor="#11112a",
   edgecolor="#333366", labelcolor="white", framealpha=0.85)

# ---------------------------------------------------------------------------
# Precomputed Moon ring template
# ---------------------------------------------------------------------------
# Semi-axes of the DISPLAY Moon ring
RING_A = A_M * MOON_SCALE                         # semi-major [AU]
RING_B = A_M * np.sqrt(1 - ECC_M_VIS**2) * MOON_SCALE  # semi-minor [AU]
RING_C = A_M * ECC_M_VIS * MOON_SCALE             # focal offset [AU]
# The ring ellipse (centred at Earth = near-focus offset from ellipse centre):
#   x_ring = earth_x - RING_C + RING_A * cos(t)
#   y_ring = earth_y            + RING_B * sin(t)

# ---------------------------------------------------------------------------
START_DATE        = datetime.date(2025, 1, 3)
AU_per_day_to_kms = 1731.456
VEL_SCALE         = 0.30 / ev_arr.max()

# ---------------------------------------------------------------------------
# Animation init (clean loop reset)
# ---------------------------------------------------------------------------
def init():
    earth_trail.set_data([], [])
    earth_dot.set_data([], [])
    vel_arrow.set_positions((0, 0), (0, 0))
    moon_ring.set_data([], [])
    em_line.set_data([], [])
    moon_trail.set_data([], [])
    moon_dot.set_data([], [])
    date_text.set_text("")
    dist_text.set_text("")
    vel_text.set_text("")
    moon_text.set_text("")
    prec_text.set_text("")
    return (earth_trail, earth_dot, vel_arrow,
            moon_ring, em_line, moon_trail, moon_dot,
            date_text, dist_text, vel_text, moon_text, prec_text)

# ---------------------------------------------------------------------------
# Animation update
# ---------------------------------------------------------------------------
def update(frame):
    i = frame_idx[frame]

    # ---- Earth ----
    ex, ey  = ex_arr[i], ey_arr[i]
    vx, vy  = evx_arr[i], evy_arr[i]
    r_sun   = er_arr[i]

    f0 = max(0, frame - EARTH_TRAIL_LEN)
    ti = frame_idx[f0: frame + 1]
    earth_trail.set_data(ex_arr[ti], ey_arr[ti])
    earth_dot.set_data([ex], [ey])
    vel_arrow.set_positions((ex, ey),
                             (ex + vx * VEL_SCALE, ey + vy * VEL_SCALE))

    # ---- Moon ----
    mx, my  = mx_disp[i], my_disp[i]
    r_moon  = mr_arr[i]

    f0m = max(0, frame - MOON_TRAIL_LEN)
    tim = frame_idx[f0m: frame + 1]
    moon_trail.set_data(mx_disp[tim], my_disp[tim])
    moon_dot.set_data([mx], [my])

    # Moon orbit ring: perifocal ellipse rotated by current ω(t)
    cos_om_i = np.cos(omega_arr[i])
    sin_om_i = np.sin(omega_arr[i])
    # Ring template in perifocal frame (focus = Earth at origin of this frame)
    rx_peri  = -RING_C + RING_A * _ring_ux
    ry_peri  =           RING_B * _ring_uy
    # Rotate into inertial frame, then translate to Earth's heliocentric position
    moon_ring.set_data(ex + rx_peri * cos_om_i - ry_peri * sin_om_i,
                       ey + rx_peri * sin_om_i + ry_peri * cos_om_i)

    em_line.set_data([ex, mx], [ey, my])

    # ---- HUD ----
    day = frame
    current_date = START_DATE + datetime.timedelta(days=day)
    date_text.set_text(
        f"Date : {current_date.strftime('%b %d, %Y')}  (day {day+1:3d}/365)"
    )
    dist_text.set_text(f"r☉   : {r_sun:.5f} AU  (Earth–Sun)")
    vel_text.set_text(
        f"v⊕   : {ev_arr[i] * AU_per_day_to_kms:.2f} km/s  (Earth)"
    )
    moon_text.set_text(
        f"r☽   : {r_moon * 149_597_870.7:.0f} km  (Earth–Moon)"
    )
    omega_deg = np.degrees(omega_arr[i]) % 360
    prec_text.set_text(f"ω☽   : {omega_deg:.1f}°  (perigee angle)")

    return (earth_trail, earth_dot, vel_arrow,
            moon_ring, em_line, moon_trail, moon_dot,
            date_text, dist_text, vel_text, moon_text, prec_text)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
anim = FuncAnimation(
    fig, update,
    init_func=init,
    frames=FRAMES_PER_ORBIT,
    interval=INTERVAL_MS,
    blit=True,
)

plt.tight_layout()
anim.save('earth_orbit.gif', writer='pillow', fps=30, dpi=80)
plt.show()
