import os
import math
from math import gcd, pi, sin, cos, atan, log, sqrt
import csv
from dataclasses import dataclass, asdict
import numpy as np
import femm

MU0 = 4e-7 * math.pi


def _lcm(a: int, b: int) -> int:
    return abs(a * b) // gcd(a, b)


# ---------------------------------------------------
# dq → abc
# ---------------------------------------------------
def dq_to_abc(id_a, iq_a, theta_e_rad):
    c = math.cos(theta_e_rad)
    s = math.sin(theta_e_rad)

    i_alpha = c*id_a - s*iq_a
    i_beta  = s*id_a + c*iq_a

    ia = i_alpha
    ib = -0.5*i_alpha + math.sqrt(3)/2*i_beta
    ic = -0.5*i_alpha - math.sqrt(3)/2*i_beta

    return ia, ib, ic


# ---------------------------------------------------
# FEMM core
# ---------------------------------------------------
def open_femm(show_gui=True):
    femm.openfemm(1 if show_gui else 0)
    femm.newdocument(0)

def load_model(fem_path):
    femm.opendocument(fem_path)

def solve():
    femm.mi_analyze(1)
    femm.mi_loadsolution()


# ---------------------------------------------------
# Sliding band
# ---------------------------------------------------
def set_sliding_band_angle(band_name, band_pos_deg):
    femm.mi_modifyboundprop(band_name, 11, band_pos_deg)

def get_torque_slidingband(band_name):
    return float(femm.mo_gapintegral(band_name, 0))


# ---------------------------------------------------
# Currents
# ---------------------------------------------------
def set_currents(circuits, ia, ib, ic):
    femm.mi_modifycircprop(circuits["A"], 1, ia)
    femm.mi_modifycircprop(circuits["B"], 1, ib)
    femm.mi_modifycircprop(circuits["C"], 1, ic)


# ---------------------------------------------------
# Sampling airgap
# ---------------------------------------------------
def sample_airgap(r_in, r_out, n_r, n_th):
    rs = np.linspace(r_in, r_out, n_r)
    th = np.linspace(0, 2*math.pi, n_th, endpoint=False)

    pts = []
    for r in rs:
        x = r*np.cos(th)
        y = r*np.sin(th)
        pts.append(np.stack([x,y], axis=1))

    return np.vstack(pts)


def export_points(csv_path, pts, include_A=True):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    with open(csv_path,"w",newline="") as f:
        w = csv.writer(f)
        header=["x","y","nu","Jz"]
        if include_A:
            header.append("Az_fem")
        w.writerow(header)

        for x,y in pts:
            row=[x,y,1/MU0,0]
            if include_A:
                Az=femm.mo_geta(x,y)
                row.append(Az)
            w.writerow(row)


# ===================================================================
# Analytical cogging torque estimation  (Zhu & Howe energy method)
#
# Reference:
#   Z.Q. Zhu and D. Howe, "Influence of design parameters on cogging
#   torque in permanent magnet machines", IEEE Trans. Energy
#   Conversion, vol. 15, no. 4, pp. 407-412, Dec. 2000.
# ===================================================================


def cogging_params(Ns: int, poles: int,
                   k1: float = 0.0, k2: float = 0.0):
    """
    Compute cogging-torque geometry parameters from the slot/pole
    combination.  Replicates the logic of PMSM._calc_cogging_params.

    Parameters
    ----------
    Ns : int    Number of stator slots.
    poles : int Number of rotor poles (= 2 * pp).
    k1, k2 : float
        Cogging design parameters (default 0 → full pole coverage).

    Returns
    -------
    Nc : int
        Number of cogging periods per mechanical revolution = LCM(Ns, poles).
    alpha_p : float
        Magnet pole-arc / pole-pitch ratio (0 .. 1].
    kso : float
        Slot-opening fraction  = Ns / Nc.
    """
    Nc = _lcm(Ns, poles)
    N = Nc / poles
    alpha_p = (N - k1) / N + k2
    while alpha_p > 1:
        k1 += 1
        alpha_p = (N - k1) / N + k2
    kso = Ns / Nc
    return Nc, alpha_p, kso


def slot_opening_width(R_g: float, g: float, Ns: int,
                       kso: float) -> float:
    """
    Slot-opening arc-length [m] at the stator bore, replicating
    PMSM.build() (pmsm.py line 1343).

    Parameters
    ----------
    R_g : float  Airgap mid-radius [m]  (= airgap_radius in designs.csv).
    g : float    Mechanical airgap thickness [m].
    Ns : int     Number of slots.
    kso : float  Slot-opening fraction (from ``cogging_params``).
    """
    R_stator_surface = R_g - g / 2
    return 2 * pi * R_stator_surface / Ns * kso


def carter_coefficient(b_so: float, tau_s: float,
                       g_prime: float) -> float:
    """
    Carter's coefficient for the effective airgap.

    Parameters
    ----------
    b_so : float    Slot-opening width [m].
    tau_s : float   Slot pitch at airgap [m]  (= 2*pi*R_g / Ns).
    g_prime : float Effective magnetic airgap [m] (= g + h_m / mu_r).

    Returns
    -------
    k_c : float   Carter coefficient  (>= 1).
    """
    u = b_so / (2 * g_prime)
    gamma = (4 / pi) * (u * atan(u) - log(sqrt(1 + u * u)))
    return tau_s / (tau_s - gamma * g_prime)


def flux_density_harmonics(Br: float, mu_r: float, h_m: float,
                           g: float, alpha_p: float,
                           kl: float = 1.0,
                           n_max: int = 15) -> np.ndarray:
    """
    Open-circuit airgap flux density Fourier coefficients for a
    surface-mounted PM machine (simplified 1-D model).

    Only odd spatial harmonics exist (n = 1, 3, 5, …).

    B_n = (4 * Br * kl) / (n * pi)  *  h_m / (h_m + mu_r * g)
          * sin(n * pi * alpha_p / 2)

    Parameters
    ----------
    Br : float       Magnet remanence [T].
    mu_r : float     Magnet relative permeability.
    h_m : float      Magnet radial thickness [m].
    g : float        Mechanical airgap [m].
    alpha_p : float  Pole-arc / pole-pitch ratio (0..1].
    kl : float       Leakage factor (default 1).
    n_max : int      Number of odd harmonics to compute.

    Returns
    -------
    B : ndarray, shape (n_max,)
        B[i] is the amplitude of the (2*i+1)-th spatial harmonic [T].
    """
    concentration = h_m / (h_m + mu_r * g)
    B = np.empty(n_max)
    for i in range(n_max):
        n = 2 * i + 1
        B[i] = (4 * Br * kl / (n * pi)) * concentration * sin(n * pi * alpha_p / 2)
    return B


def permeance_fourier(b_so: float, tau_s: float,
                      g_prime: float,
                      n_max: int = 30) -> np.ndarray:
    """
    Fourier coefficients of the relative airgap permeance function,
    using a Carter-based rectangular-slot-opening model.

    lambda(theta) = G_0  +  sum_{k>=1} G_k cos(k Ns theta)

    where
        G_0 = 1 / k_c
        G_k = -(2 delta) / (k pi) sin(k pi b_so / tau_s),  k >= 1
        delta = (k_c - 1) / k_c  *  tau_s / b_so

    Parameters
    ----------
    b_so : float    Slot-opening width [m].
    tau_s : float   Slot pitch at airgap [m].
    g_prime : float Effective magnetic airgap [m].
    n_max : int     Number of harmonics (k = 0 … n_max).

    Returns
    -------
    G : ndarray, shape (n_max + 1,)
        G[k] is the coefficient of  cos(k * Ns * theta).
    """
    kc = carter_coefficient(b_so, tau_s, g_prime)
    delta = (kc - 1) / kc * tau_s / b_so

    G = np.empty(n_max + 1)
    G[0] = 1.0 / kc
    ratio = b_so / tau_s
    for k in range(1, n_max + 1):
        G[k] = -(2 * delta) / (k * pi) * sin(k * pi * ratio)
    return G


def cogging_torque_zhu(Ns: int, poles: int,
                       Br: float, mu_r: float, h_m: float,
                       g: float, alpha_p: float, b_so: float,
                       R_g: float, L_stk: float,
                       kl: float = 1.0,
                       n_flux: int = 15, n_perm: int = 30,
                       n_theta: int = 3600,
                       n_alpha: int = 360) -> float:
    """
    Peak amplitude of the fundamental cogging torque component,
    estimated via the Zhu-Howe energy method.

    The stored energy in the airgap

        W(alpha) = (L_stk R_g g) / (2 mu_0) int_0^{2pi} B^2(theta, alpha) dtheta

    is computed on a fine angular grid for many rotor positions alpha
    over one cogging period.  The cogging torque T_cog = -dW/dalpha
    is obtained by numerical differentiation and the peak of its
    absolute value is returned.

    Parameters
    ----------
    Ns : int         Number of stator slots.
    poles : int      Number of rotor poles.
    Br : float       Magnet remanence [T].
    mu_r : float     Magnet relative permeability.
    h_m : float      Magnet thickness [m].
    g : float        Mechanical airgap [m].
    alpha_p : float  Pole-arc / pole-pitch ratio.
    b_so : float     Slot-opening width [m].
    R_g : float      Airgap mid-radius [m].
    L_stk : float    Stack (axial) length [m].
    kl : float       Leakage factor (default 1).
    n_flux : int     Odd-harmonic count for B expansion.
    n_perm : int     Harmonic count for permeance expansion.
    n_theta : int    Angular integration points (0..2pi).
    n_alpha : int    Rotor positions per cogging period.

    Returns
    -------
    T_cog_peak : float
        Peak cogging torque amplitude [N·m].
    """
    p = poles // 2
    Nc = _lcm(Ns, poles)
    g_prime = g + h_m / mu_r
    tau_s = 2 * pi * R_g / Ns

    # Fourier coefficients
    B_n = flux_density_harmonics(Br, mu_r, h_m, g, alpha_p, kl, n_flux)
    G_k = permeance_fourier(b_so, tau_s, g_prime, n_perm)

    # Angular grids
    theta = np.linspace(0, 2 * pi, n_theta, endpoint=False)
    cog_period = 2 * pi / Nc
    alphas = np.linspace(0, cog_period, n_alpha, endpoint=False)

    # Permeance profile (stator-fixed, independent of alpha)
    lam = np.full(n_theta, G_k[0])
    for k in range(1, n_perm + 1):
        lam += G_k[k] * np.cos(k * Ns * theta)

    # Vectorised energy computation: shape (n_alpha, n_theta)
    B_field = np.zeros((n_alpha, n_theta))
    theta_2d = theta[np.newaxis, :]          # (1, n_theta)
    alpha_2d = alphas[:, np.newaxis]         # (n_alpha, 1)
    for i in range(n_flux):
        n = 2 * i + 1
        B_field += B_n[i] * np.cos(n * p * (theta_2d - alpha_2d))

    B_total = B_field * lam[np.newaxis, :]   # element-wise
    dtheta = 2 * pi / n_theta

    # W(alpha) = scale * integral B^2 dtheta
    W = np.sum(B_total ** 2, axis=1) * dtheta
    scale = L_stk * R_g * g / (2 * MU0)
    W *= scale

    # T_cog = -dW / dalpha
    T_cog = -np.gradient(W, cog_period / n_alpha)

    return float(np.max(np.abs(T_cog)))


def cogging_torque_from_row(row, Br: float = 1.45, mu_r: float = 1.05,
                            kl: float = 0.98,
                            k1: float = 0.0, k2: float = 0.0) -> float:
    """
    Convenience wrapper: extract geometry from a *designs.csv* row and
    return the analytical cogging torque peak amplitude [N·m].

    ``row`` must behave like a dict with keys:
        slots, pp, magnet_thickness, airgap_thickness,
        airgap_radius, stator_length.

    b_so is recomputed from airgap_radius (the exact value stored
    by PMSM.build()), so it is guaranteed to match the motor geometry.
    alpha_p is derived from the slot/pole/k1/k2 combination.

    Parameters
    ----------
    row : dict-like   A row from designs.csv.
    Br : float        Magnet remanence [T]  (default 1.45, N52).
    mu_r : float      Magnet relative permeability (default 1.05).
    kl : float        Leakage factor (default 0.98).
    k1, k2 : float    Cogging design parameters (default 0).

    Returns
    -------
    T_cog_peak : float   Peak cogging torque [N·m], or NaN if the row
                         has invalid / missing geometry data.
    """
    try:
        Ns = int(row['slots'])
        poles = int(row['pp']) * 2
        h_m = float(row['magnet_thickness'])
        g = float(row['airgap_thickness'])
        R_g = float(row['airgap_radius'])
        L_stk = float(row['stator_length'])
    except (ValueError, KeyError):
        return float('nan')

    if any(math.isnan(v) for v in (h_m, g, R_g, L_stk)):
        return float('nan')

    Nc, alpha_p, kso = cogging_params(Ns, poles, k1, k2)
    b_so = slot_opening_width(R_g, g, Ns, kso)

    return cogging_torque_zhu(Ns, poles, Br, mu_r, h_m, g,
                              alpha_p, b_so, R_g, L_stk, kl)
