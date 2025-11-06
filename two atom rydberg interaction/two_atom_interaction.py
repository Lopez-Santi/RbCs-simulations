'''
two_atom_interaction.py
Rydberg two-atom interaction simulation with QuTiP + ARC
Simulates Na-Cs Rydberg interaction spectroscopy with thermal motion-induced broadening.
Sequence: (1) π pulse prepares Cs in |r_Cs>, (2) scan Na detuning during Na π pulse.

Date: 30 October 2025
Author: santi

'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import curve_fit
from dataclasses import dataclass
from typing import Tuple, Literal, Optional, Callable
import time

# QuTiP
from qutip import basis, qeye, sigmax, sigmaz, sigmam, tensor, mesolve, Qobj
# from qutip.qip.circuit.interpolation import Cubic_Spline # piecewise time coefficients

# ARC (install with: pip install ARC-Alkali-Rydberg-Calculator)
from arc import Sodium, Cesium, PairStateInteractions  # species and interactions

from dataclasses import dataclass, field
import numpy as np
from typing import Literal, Optional, Tuple, Callable

# Physical helpers & constants
# ----------------------------
kB = 1.380649e-23  # J/K
hbar = 1.054_571_817e-34  # J·s
TWOPI = 2.0 * np.pi

@dataclass
class ExperimentConfig:
    """
    Configuration for the two-atom interaction experiment.
    """
    # geometry: place Na at +x relative to Cs by R_mean_um
    R_mean_um: float = 5.0 # mean interatomic distance (μm)
    R_axis_um: Tuple[float, float, float] = field(default_factory=lambda: (1.0, 0.0, 0.0))  # unit vector

    OMEGA_Na_Hz: float = 5.0e5  # Na Rabi frequency (Hz)
    OMEGA_Cs_Hz: float = 0.0  # Cs Rabi frequency (Hz)
    # use default_factory for arrays
    Delta_scan_Hz: np.ndarray = field(
        default_factory=lambda: np.linspace(-10e6, 10e6, 161)
    )

    t_wait_s: float = 0.0 # wait time between pulses
    
    # Order: (x, y, z). (radial, radial, axial)
    # These are the *ground-state* trap temps of each tweezer. (uK)
    T_uK_Na: Tuple[float, float, float] = (2.0, 2.0, 2.0)
    T_uK_Cs: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    # These are the *ground-state* trap frequencies of each tweezer. (loading depth)
    omega_trap_Na_Hz: Tuple[float, float, float] = (300e3 * 0.02, 300e3 * 0.02, 50e3 * 0.02)  # multipled by drop factor of 2%
    omega_trap_Cs_Hz: Tuple[float, float, float] = (100e3 * 0.02, 100e3 * 0.02, 20e3 * 0.02)  # multipled by drop factor of 2%

    T2_Na_s: float = 5e-6 # coherence time for Na
    T2_Cs_s: float = 20e-6 # coherence time for Cs
    T1_use_ARC: bool = True # whether to use ARC lifetimes for T1

    broadening: bool = True  # turn on or off broadening

    N_mc: int = 200  # number of Monte Carlo samples per detuning
    N_steps: int = 20  # number of time steps in evolution
    seed: Optional[int] = 1234

    n_Na: int = 51 # Rydberg principal quantum number for Na
    n_Cs: int = 54 # Rydberg principal quantum number for Cs
    l_Na: int = 0 # Rydberg orbital quantum number for Na
    l_Cs: int = 0 # Rydberg orbital quantum number for Cs
    j_Na: float = 0.5 # Rydberg total angular momentum for Na
    j_Cs: float = 0.5 # Rydberg total angular momentum for Cs

    load_depth_T_Na: float = 300 # uK
    load_depth_T_Cs: float = 100 # uK

    load_factor: float = np.sqrt(0.01)  # % of trap depth after dropping
    
    plot_displacements: bool = False # whether to plot displacement histograms after simulation
    displacement_list: list = field(default_factory=list) # to store displacements for plotting

    mass_Na: float = Sodium().mass # mass of Na atom (kg)
    mass_Cs: float = Cesium().mass # mass of Cs atom (kg)

# ----------------------------
# C6 calculation with ARC - using perturbation theory
# ----------------------------
def compute_c6_na_cs_rad_per_s_um6(cfg: ExperimentConfig,
                                   theta: float = 0.0, phi: float = 0.0,
                                   nRange: int = 5, lRange: int = 5,
                                   deltaMax_Hz: float = 25e9,
                                   r_range_um: Tuple[float, float] = (3.0, 10.0),
                                   n_eigs: int = 250,
                                   progress: bool = False) -> float:
    """
    Use ARC's PairStateInteractions to find C6 for Na(51S1/2)-Cs(54S1/2).
    Returns C6 in angular units: [rad/s * μm^6] (so V(t) = C6 / R(t)^6).
    """
    # Build inter-species pair-state calculation
    calc = PairStateInteractions(
        Sodium(), cfg.n_Na, cfg.l_Na, cfg.j_Na,
        cfg.n_Cs, cfg.l_Cs, cfg.j_Cs,
        m1=0.5, m2=0.5, atom2=Cesium()
    )
    # calculate C6 using getC6perturbatively
    c6_freq_um6 = calc.getC6perturbatively(theta=theta, phi=phi, nRange=nRange, energyDelta=deltaMax_Hz)
    print(f"Computed C6: {c6_freq_um6} GHz·μm^6")
    c6_Hz_um6 = c6_freq_um6 * 1.0e9  # GHz -> Hz if needed by ARC version
    c6_rad_um6 = TWOPI * c6_Hz_um6   # Hz -> rad/s
    return (-1)*c6_rad_um6 # multiply by -1 to get correct sign (ARC gives V = -C6 / R^6)

#############################
### Functions for simulation.
##############################

# State lifetimes from ARC
# ----------------------------
def get_state_lifetimes_from_arc(
    temperature_Na_K: float = 300.0,
    temperature_Cs_K: float = 300.0,
    n_l_j_Na=(51,0,0.5),
    n_l_j_Cs=(54,0,0.5),
    n_upper_offset: int = 25,
):
    """
    Get T1 lifetimes for Na and Cs Rydberg states from ARC.
    Parameters:
        temperature_Na_K: temperature for Na (K)
        temperature_Cs_K: temperature for Cs (K)
        n_l_j_Na: tuple of (n, l, j) for Na Rydberg state
        n_l_j_Cs: tuple of (n, l, j) for Cs Rydberg state
        n_upper_offset: include levels up to n + n_upper_offset for blackbody
    Returns: (T1_Na [s], T1_Cs [s])
    """
    Na = Sodium()
    Cs = Cesium()
    nNa = n_l_j_Na[0]
    nCs = n_l_j_Cs[0]

    T1_Na = Na.getStateLifetime(
        *n_l_j_Na,
        temperature=temperature_Na_K,
        includeLevelsUpTo=nNa + n_upper_offset
    )
    T1_Cs = Cs.getStateLifetime(
        *n_l_j_Cs,
        temperature=temperature_Cs_K,
        includeLevelsUpTo=nCs + n_upper_offset
    )
    return float(T1_Na), float(T1_Cs)

# Hamiltonian construction
# ----------------------------
def two_atom_ops():
    """
    Projectors and Pauli ops on a 2x2 two-level Hilbert space (Na * Cs).
    Returns: (n_Na, n_Cs, n_Na_n_Cs, sx_Na, sx_Cs, sz_Na, sz_Cs, sm_Na, sm_Cs)
    """
    g = basis(2, 0); r = basis(2, 1)
    n_single = r * r.dag()
    I = qeye(2)
    # Number operators
    n_Na = tensor(n_single, I)
    n_Cs = tensor(I, n_single)
    n_Na_n_Cs = tensor(n_single, n_single)
    # Drives (σ_x)
    sx = sigmax()
    sx_Na = tensor(sx, I)
    sx_Cs = tensor(I, sx)
    # σz (for dephasing)
    sz = sigmaz()
    sz_Na = tensor(sz, I)
    sz_Cs = tensor(I, sz)
    # Lowering
    sm = sigmam()
    sm_Na = tensor(sm, I)
    sm_Cs = tensor(I, sm)
    return n_Na, n_Cs, n_Na_n_Cs, sx_Na, sx_Cs, sz_Na, sz_Cs, sm_Na, sm_Cs

def build_time_dependent_H(OMEGA_Na_rad: float,
                           OMEGA_Cs_rad: float,
                           Delta_Na_rad: float,
                           V_of_t: Callable[[np.ndarray], np.ndarray],
                           tlist: np.ndarray) -> list:
    """
    QuTiP list-style Hamiltonian with time-dependent interaction V(t).
    H = (Ω_Na/2) sigma_x_Na + (Ω_Cs/2) sigma_x_Cs - Δ_Na n_Na + V(t) n_Na n_Cs
    Parameters:
        OMEGA_Na_rad: Na Rabi frequency (rad/s)
        OMEGA_Cs_rad: Cs Rabi frequency (rad/s)
        Delta_Na_rad: Na detuning (rad/s)
        V_of_t: function that takes np.ndarray times and returns V(t) array
        tlist: np.ndarray of times (s)
    Returns: list-style Hamiltonian for QuTiP
    """
    n_Na, _, n_Na_n_Cs, sx_Na, sx_Cs, *_ = two_atom_ops()

    H = []
    # Constant drive terms (coefficients are constants -> provide numeric prefactors)
    if OMEGA_Na_rad != 0.0:
        H.append([sx_Na, lambda t, args: 0.5 * OMEGA_Na_rad])
    if OMEGA_Cs_rad != 0.0:
        H.append([sx_Cs, lambda t, args: 0.5 * OMEGA_Cs_rad]) # no Cs drive here

    # Static Na detuning term: -Δ_Na * n_Na
    if Delta_Na_rad != 0.0:
        H.append([n_Na, lambda t, args: -Delta_Na_rad])

    # Time-dependent interaction: +V(t) * n_Na n_Cs
    V_vals = V_of_t(tlist)
    # QuTiP will linearly interpolate time-dependent arrays
    H.append([n_Na_n_Cs, V_vals])

    return H

# function to check if atom is recaptured
# ----------------------------
def is_atom_recaptured(x0: Tuple[float, float, float],
                       v0: Tuple[float, float, float],
                       m_kg: float,
                       omegas_Hz_xyz: Tuple[float, float, float],
                       Temps_uK_xyz: Tuple[float, float, float],
                       load_depth_T_uK: float,
                       tlist: np.ndarray,
                       rng) -> bool:
    """
    Check if the atom is recaptured after time tlist[-1].
    Parameters:
        x0: initial position vector (μm)
        v0: initial velocity vector (μm/s)
        m_kg: mass of the atom (kg)
        omegas_Hz_xyz: trap frequencies for each axis (Hz)
        Temps_uK_xyz: trap temperatures for each axis (μK)
        load_depth_T_uK: trap depth (μK)
        tlist: time list (s)
        rng: random number generator
    Returns: True if recaptured, False if lost.
    """
    # trap depth energy in J
    U0 = kB * load_depth_T_uK * 1e-6 # in J

    # kinetic energy of atom at end of tlist in J
    KE = 0.5 * m_kg * (np.linalg.norm(v0 * 1e-6)) ** 2 # in J

    # potential energy of atom at end of tlist in J
    r_vec = (x0 + v0 * tlist[-1]) * 1e-6 # position at end of tlist in m

    # calculate potential energy using harmonic approx
    # Ur = 0.5 * m_kg * np.sum((np.array(omegas_Hz_xyz) * TWOPI)**2 * np.array(r_vec)**2) # in J

    # calculate potential energy using gaussian beam
    
    # calculate beam waist from trap temp and radial trap freq
    w0 = np.sqrt( (2 * kB * load_depth_T_uK * 1e-6) / (m_kg * (TWOPI * omegas_Hz_xyz[0])**2) )
    
    # calculate the beam waist at position z
    w_z = w0 * np.sqrt(1 + (r_vec[2] / (np.pi * w0**2 / (616e-9)))**2 )

    # calculate potential energy
    Ur = kB * load_depth_T_uK * 1e-6 * (1 - (w0**2 / w_z**2) * np.exp(-2 * (r_vec[0]**2 + r_vec[1]**2) / w_z**2)) # in J

    # total energy in J
    total_E = KE + Ur

    return total_E <= U0

# Stochastic kinematics models
# ----------------------------

# ---------- QHO thermal initial condition sampler ----------
def _sample_axis_QHO(m_kg: float, omega_Hz: float, T_uK: float, rng) -> Tuple[float, float]:
    """
    Return (x0 [m], v0 [m/s]) for one axis from the QHO recipe (Lee Liu thesis).
    Parameters:
        m_kg: mass of the atom (kg)
        omega_Hz: trap frequency for the axis (Hz)
        T_uK: trap temperature for the axis (μK)
        rng: random number generator
    """
    temp = T_uK * 1e-6
    omega = TWOPI * omega_Hz # angular frequency in rad/s
    if temp <= 0:
        nbar = 0.0
    else:
        x = hbar * omega / (kB * temp)
        try:
            # prevent overflow at very low T
            nbar = 1.0 / (np.exp(x) - 1.0)
        except OverflowError:
            nbar = 0.0
            print("Warning: Overflow in nbar calculation; setting nbar=0. Check T and omega.")

    # geometric: P(n) = (1/(1+nbar)) * (nbar/(1+nbar))^n , mean=nbar
    p = 1.0 / (1.0 + nbar) # success probability
    n = rng.geometric(p) - 1  # sample geometric distribution for n

    E = (n + 0.5) * hbar * omega # energy in J
    theta = rng.uniform(0.0, 2*np.pi) # random phase

    x0 = np.sqrt(2*E/(m_kg * omega**2)) * np.sin(theta) # meters
    v0 = np.sqrt(2*E/(m_kg)) * np.cos(theta) # m/s

    return x0, v0

def sample_QHO_initial_3d(m_kg: float,
                          omegas_Hz_xyz: Tuple[float, float, float],
                          Temps_uK_xyz: Tuple[float, float, float],
                          rng) -> Tuple[np.ndarray, np.ndarray]:
    """
    Uses the QHO thermal initial condition sampler for each axis.
    Parameters:
        m_kg: mass of the atom (kg)
        omegas_Hz_xyz: trap frequencies for each axis (Hz)
        Temps_uK_xyz: trap temperatures for each axis (μK)
        rng: random number generator
    Returns: (x0_vec [μm], v0_vec [m/s]) for 3 axes.
    """
    x0 = np.zeros(3); v0 = np.zeros(3)
    for i, (om, temp) in enumerate(zip(omegas_Hz_xyz, Temps_uK_xyz)):
        xi, vi = _sample_axis_QHO(m_kg, om, temp, rng)
        x0[i], v0[i] = xi * 1e6, vi * 1e6 # position in μm and velocity in μm/s
    return x0, v0

# # Lists to store initial conditions and displacements (for analysis/debugging)
# x_Cs_list = []
# x_Na_list = []
# v_Cs_list = []
# v_Na_list = []
# displacement_list = []

def make_V_of_t_generator(c6_rad_um6: float,
                          cfg: ExperimentConfig,
                          tlist: np.ndarray,
                          rng: np.random.Generator,
                          displacement_list: list) -> callable:
    """
    Build V(t) = C6 / |R(t)|^6 using QHO thermal initial conditions
    for *each species and axis* (using Lee Liu thesis section C.2).
    Parameters:
        c6_rad_um6: C6 coefficient in rad/s·μm^6
        cfg: ExperimentConfig
        tlist: np.ndarray of times (s)
        rng: np.random.Generator
    Returns:
        V_of_t: function that takes np.ndarray times and returns V(t) array
        recaptured: bool, whether Na atom is recaptured at end of tlist
    """

    # Mean axis separation vector in μm
    # make sure its normalized
    R_axis_unit = np.array(cfg.R_axis_um, dtype=float)
    R_axis_unit = R_axis_unit / np.linalg.norm(R_axis_unit)
    # multiply by mean distance
    R0_vec_um = R_axis_unit * cfg.R_mean_um
    # print(f"Mean interatomic vector R0: {R0_vec_um} μm")

    # No broadening: constant V
    if not cfg.broadening:
        def V_of_t(_times: np.ndarray) -> np.ndarray:
            R_norm = np.linalg.norm(R0_vec_um)
            V_const = c6_rad_um6 / (R_norm ** 6)
            return V_const * np.ones_like(_times)
        return V_of_t, True # broadening off, always recaptured
    
    # Sample initial (x0, v0) for each atom from QHO distribution
    m_Na = cfg.mass_Na
    m_Cs = cfg.mass_Cs

    x0_Na_um, v0_Na_umps = sample_QHO_initial_3d(
        m_Na, np.array(cfg.omega_trap_Na_Hz) * cfg.load_factor, cfg.T_uK_Na, rng
    )

    x0_Cs_um, v0_Cs_umps = sample_QHO_initial_3d(
        m_Cs, np.array(cfg.omega_trap_Cs_Hz) * cfg.load_factor, cfg.T_uK_Cs, rng
    )

    # for debugging / analysis, store sampled values
    # x_Na_list.append(x0_Na_um)
    # v_Na_list.append(v0_Na_umps)
    # x_Cs_list.append(x0_Cs_um)
    # v_Cs_list.append(v0_Cs_umps)

    # Relative position over time (μm)
    # R(t) = R0 + (x0_Na - x0_Cs) + (vNa - vCs)*t
    dx = (x0_Na_um - x0_Cs_um)
    dv = (v0_Na_umps - v0_Cs_umps)
    R_vec_t = R0_vec_um + dx + dv * tlist[:, None]
    # print(f"Initial relative position dx: {dx} μm")
    R_norm_t = np.linalg.norm(R_vec_t, axis=1)
    displacement_list.append(R_norm_t[-1])

    # avg_x_um = (np.linalg.norm(x0_Na_um - x0_Cs_um))
    # avg_v_umps = (np.linalg.norm(v0_Na_umps - v0_Cs_umps))
    # print(f"Sampled avg initial separation: {avg_x_um:.2f} μm")
    # print(f"Sampled avg relative velocity: {avg_v_umps:.2f} μm/s")
    V_t = c6_rad_um6 / (R_norm_t ** 6)

    def V_of_t(_times: np.ndarray) -> np.ndarray:
        # QuTiP will pass the same tlist; return precomputed array
        return V_t
    
    # check if Na atom is recaptured (we are dominated by Na loss)
    m_Na = cfg.mass_Na
    recaptured = is_atom_recaptured(x0_Na_um, v0_Na_umps,
        m_kg=m_Na,
        omegas_Hz_xyz=cfg.omega_trap_Na_Hz,
        Temps_uK_xyz=cfg.T_uK_Na,
        load_depth_T_uK=cfg.load_depth_T_Na,
        tlist=tlist,
        rng=rng
    ) # True if recaptured, False if lost
    
    # print(V_t)
    return V_of_t, recaptured

# Single-shot simulation
# ----------------------------
def simulate_shot(Delta_Na_Hz,
                  cfg,
                  c6_rad_um6,
                  rng,
                  T1_cache=None,
                  interaction_on=True,
                  displacement_list=[]):
    """
    Simulate one 'shot' of the Na π-pulse.
    If interaction_on=False, the V(R) term is set to zero.
    Parameters:
        Delta_Na_Hz: Na detuning for this shot (Hz)
        cfg: ExperimentConfig
        c6_rad_um6: precomputed C6 coefficient
        rng: np.random.Generator
        T1_cache: optional tuple of (T1_Na, T1_Cs) to avoid recomputing lifetimes
        interaction_on: whether to include interaction term in Hamiltonian
    Returns: P_excitation (float)
    """
    OMEGA_Na_rad = 2 * np.pi * cfg.OMEGA_Na_Hz
    OMEGA_Cs_rad = 2 * np.pi * cfg.OMEGA_Cs_Hz
    Delta_Na_rad = 2 * np.pi * Delta_Na_Hz

    # π-pulse time and time list
    t_pi = np.pi / max(OMEGA_Na_rad, 1e-30) # avoid div by zero [sec]
    tlist = np.linspace(0.0, t_pi, cfg.N_steps) # 200 time steps

    # Prepare interaction function
    if interaction_on:
        V_of_t, recaptured = make_V_of_t_generator(
            c6_rad_um6=c6_rad_um6,
            cfg=cfg,
            tlist=tlist,
            rng=rng,
            displacement_list=displacement_list
        )
        if not recaptured:
            # Atom lost; return 0 excitation probability immediately
            return 0.0
    else:
        # No interaction -> V(t) = 0 for all times
        V_of_t = lambda t: np.zeros_like(t)

    # if atom is recaptured, proceed to build Hamiltonian and simulate

    # Build Hamiltonian
    H = build_time_dependent_H(
        OMEGA_Na_rad=OMEGA_Na_rad,
        OMEGA_Cs_rad=OMEGA_Cs_rad,
        Delta_Na_rad=Delta_Na_rad,
        V_of_t=V_of_t,
        tlist=tlist,
    )

    # Initial state |g_Na, r_Cs>
    g = basis(2, 0)
    r = basis(2, 1)
    psi0 = tensor(g, r)

    # Collapses (use cached lifetimes)
    if T1_cache is not None:
        T1_Na, T1_Cs = T1_cache
    else:
        T1_Na, T1_Cs = 50e-6, 80e-6

    n_Na, n_Cs, _, _, _, sz_Na, sz_Cs, sm_Na, sm_Cs = two_atom_ops()
    c_ops = []
    if T1_Na > 0:
        c_ops.append(np.sqrt(1.0 / T1_Na) * sm_Na)
    if T1_Cs > 0:
        c_ops.append(np.sqrt(1.0 / T1_Cs) * sm_Cs)
    # Dephasing rates
    gamma_phi_Na = max(0.0, (1.0 / cfg.T2_Na_s) - 0.5 * (0.0 if T1_Na == 0 else 1.0 / T1_Na))
    gamma_phi_Cs = max(0.0, (1.0 / cfg.T2_Cs_s) - 0.5 * (0.0 if T1_Cs == 0 else 1.0 / T1_Cs))
    if gamma_phi_Na > 0:
        c_ops.append(np.sqrt(0.5 * gamma_phi_Na) * sz_Na)
    if gamma_phi_Cs > 0:
        c_ops.append(np.sqrt(0.5 * gamma_phi_Cs) * sz_Cs)

    result = mesolve(H, psi0, tlist, c_ops=c_ops, e_ops=[n_Na])
    
    return 1-float(np.real(result.expect[0][-1]))

# Detuning scan (Monte Carlo)
# ----------------------------
def scan_detuning(cfg, c6_rad_um6=None, interaction_on=True):
    """
    Scan over Na detunings with Monte Carlo sampling.
    Parameters:
        cfg: ExperimentConfig
        c6_rad_um6: optional precomputed C6 coefficient
        interaction_on: whether to include interaction term in Hamiltonian
    Returns: (Delta_scan_Hz, P_excitation)
    """
    rng = np.random.default_rng(cfg.seed)
    if c6_rad_um6 is None:
        c6_rad_um6 = compute_c6_na_cs_rad_per_s_um6(cfg)

    if cfg.T1_use_ARC:
        T1_Na, T1_Cs = get_state_lifetimes_from_arc(
            temperature_Na_K=np.mean(cfg.T_uK_Na) * 1e-6,
            temperature_Cs_K=np.mean(cfg.T_uK_Cs) * 1e-6,
            n_l_j_Na=(cfg.n_Na, cfg.l_Na, cfg.j_Na),
            n_l_j_Cs=(cfg.n_Cs, cfg.l_Cs, cfg.j_Cs),
            n_upper_offset=0, # arbitrary cutoff for levels included in ARC lifetime calc
        )

    else:
        T1_Na, T1_Cs = 0, 0

    if interaction_on & cfg.broadening:
        cfg.displacement_list = []  # to store displacements for analysis

    # for debugging
    print("Starting detuning scan...")
    P = np.zeros_like(cfg.Delta_scan_Hz)
    for i, dHz in enumerate(cfg.Delta_scan_Hz):
        if i == 0:
            start_time = time.time()
        if (i+1) % 2 == 0:
            elapsed = time.time() - start_time
            avg_time_per_point = elapsed / (i+1)
            remaining_points = len(cfg.Delta_scan_Hz) - (i+1)
            est_remaining = avg_time_per_point * remaining_points
            print(f"  Point {i+1}/{len(cfg.Delta_scan_Hz)} - "
                  f"Elapsed: {elapsed:.1f}s, "
                  f"Est. remaining: {est_remaining:.1f}s")
        acc = 0.0
        for _ in range(cfg.N_mc):
            acc += simulate_shot(
                dHz, cfg, c6_rad_um6, rng, T1_cache=(T1_Na, T1_Cs),
                interaction_on=interaction_on,
                displacement_list=cfg.displacement_list
            )
        P[i] = acc / cfg.N_mc # average over MC shots
    
    # plot average displacement for debugging
    if cfg.plot_displacements:
        # plot histogram of displacement_list
        plt.figure(figsize=(8,5))
        all_displacements = np.concatenate(cfg.displacement_list)
        plt.hist(all_displacements, bins=50, density=True, alpha=0.7, color='blue')
        plt.xlabel("Interatomic distance |R| (μm)")
        plt.ylabel("Probability density")
        plt.title("Histogram of interatomic distances during interaction")
        plt.axvline(cfg.R_mean_um, color='red', linestyle='--', label='Mean distance')
        plt.legend()
        plt.tight_layout()
        plt.show()
    return cfg.Delta_scan_Hz, P


###############################
### fitting and plotting utils
###############################
def lorentzian(x, A, x0, gamma):
    return 1 - A * (gamma**2 / ((x - x0)**2 + gamma**2))