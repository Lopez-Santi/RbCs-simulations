'''
two_atom_interaction.py
Rydberg two-atom interaction simulation with QuTiP + ARC
Simulates any two atom Rydberg interaction spectroscopy with thermal motion-induced broadening.
Sequence: (1) π pulse prepares atom2 in |r_1>, (2) scan atom1 detuning during atom1 π pulse.

Date Created: 06 November 2025
Last Modified: 03 December 2025
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
from arc import PairStateInteractions, AlkaliAtom  # species and interactions'
# import all the atoms we might use
from arc import Sodium, Cesium, Rubidium, Potassium

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
    ### General atom information ###
    # atom1 information - always the atom you are measuring the spectrum of
    atom1: AlkaliAtom = field(default_factory=lambda: Sodium())
    mass_atom1: float = Sodium().mass # mass of atom1 (kg)
    n_atom1: int = 51 # Rydberg principal quantum number for atom1
    l_atom1: int = 0 # Rydberg orbital quantum number for atom1
    j_atom1: float = 0.5 # Rydberg total angular momentum for atom1
    m1: int = 0.5 # magnetic quantum number for atom1
    # Rabi frequency and coherence time
    OMEGA_atom1_Hz: float = 5.0e5  # atom1 Rabi frequency (Hz)
    T2_atom1_s: float = 5e-6 # coherence time for atom1
    wavelength_nm_atom1: float = 616.0 # trapping wavelength for atom1 (nm)

    # atom2 information - always the atom that is excited first
    atom2: AlkaliAtom = field(default_factory=lambda: Cesium())
    mass_atom2: float = Cesium().mass # mass of atom2 (kg)
    n_atom2: int = 54 # Rydberg principal quantum number for atom2
    l_atom2: int = 0 # Rydberg orbital quantum number for atom2
    j_atom2: float = 0.5 # Rydberg total angular momentum for atom2
    m2: int = 0.5 # magnetic quantum number for atom2
    # Rabi frequency and coherence time
    OMEGA_atom2_Hz: float = 0.0  # atom2 Rabi frequency (Hz) (set to 0 always)
    T2_atom2_s: float = 20e-6 # coherence time for atom2
    wavelength_nm_atom2: float = 1064.0 # trapping wavelength for atom2 (nm)
    
    ### Geometry and timing ###
    # geometry: place atom1 at +x relative to atom2 by R_mean_um
    R_mean_um: float = 5.0 # mean interatomic distance (μm)
    R_axis_um: Tuple[float, float, float] = field(default_factory=lambda: (1.0, 0.0, 0.0))  # unit vector

    # relevant timing settings
    t_pi_atom1_us: float = None # atom1 π pulse time (μs)
    # for just simulating release recapture t_pi_atom2_us must be set to 0 - otherwise it will be taken into account when calculating R(t)
    t_pi_atom2_us: float = 0.0 # atom2 π pulse time (μs) (should be the pulse time used for preparing atom2 in |r>)
    t_wait_s: float = 0.0 # wait time between pulses
    T1_use_ARC: bool = True # whether to use ARC lifetimes for T1

    ### Trap parameters for both atoms ###
    # Order: (x, y, z). (radial, radial, axial)
    # These are the *ground-state* trap temps of each tweezer. (uK)
    T_uK_atom1: Tuple[float, float, float] = (2.0, 2.0, 2.0)
    T_uK_atom2: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    # These are the *ground-state* trap frequencies of each tweezer. (loading depth)
    omega_trap_atom1_Hz: Tuple[float, float, float] = (300e3 * 0.02, 300e3 * 0.02, 50e3 * 0.02)  # multipled by drop factor of 2%
    omega_trap_atom2_Hz: Tuple[float, float, float] = (100e3 * 0.02, 100e3 * 0.02, 20e3 * 0.02)  # multipled by drop factor of 2%
    # depth of the trap when loaded (uK)
    load_depth_T_atom1: float = 300 # uK
    load_depth_T_atom2: float = 100 # uK

    load_factor: float = np.sqrt(0.01)  # % of trap depth after dropping
    squeeze_factor: Tuple[float, float, float] = (1.0, 1.0, 1.0) # factor to squeeze position/momentum spread (1.0 = no squeezing) (x = x / squeeze_factor, p = p * squeeze_factor)

    ### simulation information ###
    # use default_factory for arrays
    Delta_scan_Hz: np.ndarray = field(
        default_factory=lambda: np.linspace(-10e6, 10e6, 161)
    )

    broadening: bool = True  # turn on or off broadening
    simulate_atom1_recapture: bool = False  # whether to simulate atom1 recapture
    simulate_atom2_recapture: bool = False  # whether to simulate atom2 recapture
    hamiltonian_on: bool = True  # whether to turn on the Hamiltonian evolution
    
    N_mc: int = 200  # number of Monte Carlo samples per detuning
    N_steps: int = 20  # number of time steps in evolution
    seed: Optional[int] = None#1234
    
    plot_displacements: bool = False # whether to plot displacement histograms after simulation
    positions_list: list = field(default_factory=list) # to store positions for plotting
    velocities_list: list = field(default_factory=list) # to store velocities for plotting
    energy_array1: list = field(default_factory=list) # to store energy arrays if recapture simulated
    energy_array2: list = field(default_factory=list) # to store energy arrays if recapture simulated
    positions1_list: list = field(default_factory=list) # to store positions if recapture simulated
    positions2_list: list = field(default_factory=list) # to store positions if recapture simulated
    velocities1_list: list = field(default_factory=list) # to store velocities if recapture simulated
    velocities2_list: list = field(default_factory=list) # to store velocities if recapture simulated
    n_list: list = field(default_factory=list) # to store n values if needed

# ----------------------------
# C6 calculation with ARC - using perturbation theory
# ----------------------------
def compute_c6_atom1_atom2_rad_per_s_um6(cfg: ExperimentConfig,
                                   theta: float = 0.0, phi: float = 0.0,
                                   nRange: int = 5, lRange: int = 5,
                                   deltaMax_Hz: float = 25e9,
                                   r_range_um: Tuple[float, float] = (3.0, 10.0),
                                   n_eigs: int = 250,
                                   progress: bool = False) -> float:
    """
    Use ARC's PairStateInteractions to find C6 for atom1 + atom2.
    Returns C6 in angular units: [rad/s * μm^6] (so V(t) = C6 / R(t)^6).
    """
    # Build inter-species pair-state calculation
    calc = PairStateInteractions(
        cfg.atom1, cfg.n_atom1, cfg.l_atom1, cfg.j_atom1,
        cfg.n_atom2, cfg.l_atom2, cfg.j_atom2,
        m1=cfg.m1, m2=cfg.m2, atom2=cfg.atom2
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
def get_state_lifetimes_from_arc(cfg: ExperimentConfig,
    # temperature_atom1_K: float = 300.0,
    # temperature_atom2_K: float = 300.0,
    # n_l_j_atom1=(51,0,0.5),
    # n_l_j_atom2=(54,0,0.5),
    n_upper_offset: int = 25,
):
    """
    Get T1 lifetimes for atom1 and atom2 Rydberg states from ARC.
    Parameters:
        # temperature_atom1_K: temperature for atom1 (K)
        # temperature_atom2_K: temperature for atom2 (K)
        # n_l_j_atom1: tuple of (n, l, j) for atom1 Rydberg state
        # n_l_j_atom2: tuple of (n, l, j) for atom2 Rydberg state
        n_upper_offset: include levels up to n + n_upper_offset for blackbody (hardcoded to 0)
    Returns: (T1_atom1 [s], T1_atom2 [s])
    """
    atom1 = cfg.atom1
    atom2 = cfg.atom2
    n_l_j_atom1 = (cfg.n_atom1, cfg.l_atom1, cfg.j_atom1)
    n_l_j_atom2 = (cfg.n_atom2, cfg.l_atom2, cfg.j_atom2)
    n_atom1 = cfg.n_atom1
    n_atom2 = cfg.n_atom2

    T1_atom1 = atom1.getStateLifetime(
        *n_l_j_atom1,
        temperature= np.mean(cfg.T_uK_atom1) * 1e-6, # convert μK to K
        includeLevelsUpTo=n_atom1 + n_upper_offset
    )
    T1_atom2 = atom2.getStateLifetime(
        *n_l_j_atom2,
        temperature= np.mean(cfg.T_uK_atom2) * 1e-6, # convert μK to K
        includeLevelsUpTo=n_atom2 + n_upper_offset
    )
    return float(T1_atom1), float(T1_atom2)

# Hamiltonian construction
# ----------------------------
def two_atom_ops():
    """
    Projectors and Pauli ops on a 2x2 two-level Hilbert space (atom1 * atom2).
    Returns: (n_atom1, n_atom2, n_atom1_n_atom2, sx_atom1, sx_atom2, sz_atom1, sz_atom2, sm_atom1, sm_atom2)
    """
    g = basis(2, 0); r = basis(2, 1)
    n_single = r * r.dag()
    I = qeye(2)
    # Number operators
    n_atom1 = tensor(n_single, I)
    n_atom2 = tensor(I, n_single)
    n_atom1_n_atom2 = tensor(n_single, n_single)
    # Drives (σ_x)
    sx = sigmax()
    sx_atom1 = tensor(sx, I)
    sx_atom2 = tensor(I, sx)
    # σz (for dephasing)
    sz = sigmaz()
    sz_atom1 = tensor(sz, I)
    sz_atom2 = tensor(I, sz)
    # Lowering
    sm = sigmam()
    sm_atom1 = tensor(sm, I)
    sm_atom2 = tensor(I, sm)
    return n_atom1, n_atom2, n_atom1_n_atom2, sx_atom1, sx_atom2, sz_atom1, sz_atom2, sm_atom1, sm_atom2

def build_time_dependent_H(OMEGA_atom1_rad: float,
                           OMEGA_atom2_rad: float,
                           Delta_atom1_rad: float,
                           V_of_t: Callable[[np.ndarray], np.ndarray],
                           tlist: np.ndarray) -> list:
    """
    QuTiP list-style Hamiltonian with time-dependent interaction V(t).
    H = (Ω_atom1/2) sigma_x_atom1 + (Ω_atom2/2) sigma_x_atom2 - Δ_atom1 n_atom1 + V(t) n_atom1 n_atom2
    Parameters:
        OMEGA_atom1_rad: atom1 Rabi frequency (rad/s)
        OMEGA_atom2_rad: atom2 Rabi frequency (rad/s)
        Delta_atom1_rad: atom1 detuning (rad/s)
        V_of_t: function that takes np.ndarray times and returns V(t) array
        tlist: np.ndarray of times (s)
    Returns: list-style Hamiltonian for QuTiP
    """
    n_atom1, n_atom2, n_atom1_n_atom2, sx_atom1, sx_atom2, *_ = two_atom_ops()

    H = []
    # Constant drive terms (coefficients are constants -> provide numeric prefactors)
    if OMEGA_atom1_rad != 0.0:
        H.append([sx_atom1, lambda t, args: 0.5 * OMEGA_atom1_rad])
    if OMEGA_atom2_rad != 0.0:
        H.append([sx_atom2, lambda t, args: 0.5 * OMEGA_atom2_rad]) # no atom2 drive here

    # Static atom1 detuning term: -Δ_atom1 * n_atom1
    if Delta_atom1_rad != 0.0:
        H.append([n_atom1, lambda t, args: -Delta_atom1_rad])

    # Time-dependent interaction: +V(t) * n_atom1 n_atom2
    V_vals = V_of_t(tlist)
    # QuTiP will linearly interpolate time-dependent arrays
    H.append([n_atom1_n_atom2, V_vals])

    return H

# function to check if atom is recaptured
# ----------------------------
def is_atom_recaptured(cfg: ExperimentConfig,
                       x0: Tuple[float, float, float],
                       v0: Tuple[float, float, float],
                       m_kg: float,
                       omegas_Hz_xyz: Tuple[float, float, float],
                       Temps_uK_xyz: Tuple[float, float, float],
                       load_depth_T_uK: float,
                       tlist: np.ndarray,
                       wavelength_nm: float,
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
    # calculate beam waist from radial and axial trap frequencies
    # w0 = (omegas_Hz_xyz[0] / omegas_Hz_xyz[2]) * wavelength_nm*1e-9 / (np.pi * np.sqrt(2))
    w0 = 0.6*1e-6 # hardcoded beam waist in m
    # print(w0)
    # calculate the beam waist at position z
    w_z = w0 * np.sqrt(1 + (r_vec[2] / (np.pi * w0**2 / (wavelength_nm*1e-9)))**2 )

    # calculate potential energy
    Ur = kB * load_depth_T_uK * 1e-6 * (1 - (w0**2 / w_z**2) * np.exp(-2 * (r_vec[0]**2 + r_vec[1]**2) / w_z**2)) # in J

    # total energy in J
    total_E = KE + Ur
    energy_array = np.array([KE, Ur, U0])
    return total_E <= U0, energy_array

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

    # mean occupation number
    nbar = 1.0 / (np.exp(hbar * omega / (kB * temp)) - 1.0)
    # print(temp, omega / (2.0 * np.pi), nbar)
    # print(nbar)
    # geometric: P(n) = (1/(1+nbar)) * (nbar/(1+nbar))^n , mean=nbar
    p = 1.0 / (1.0 + nbar) # success probability
    n = rng.geometric(p) - 1  # sample geometric distribution for n
    # print(n)
    E = (n + 0.5) * hbar * omega # energy in J
    theta = rng.uniform(0.0, 2*np.pi) # random phase

    x0 = np.sqrt(2*E/(m_kg * omega**2)) * np.sin(theta) # meters
    v0 = np.sqrt(2*E/(m_kg)) * np.cos(theta) # m/s

    return x0, v0, nbar

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
    x0 = np.zeros(3); v0 = np.zeros(3); n = np.zeros(3)
    for i, (om, temp) in enumerate(zip(omegas_Hz_xyz, Temps_uK_xyz)):
        xi, vi, ni = _sample_axis_QHO(m_kg, om, temp, rng)
        x0[i], v0[i], n[i] = xi * 1e6, vi * 1e6, ni # position in μm and velocity in μm/s
    return x0, v0, n



def make_V_of_t_generator(c6_rad_um6: float,
                          cfg: ExperimentConfig,
                          tlist: np.ndarray,
                          rng: np.random.Generator,
                          positions_list: list,
                          velocities_list: list) -> callable:
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
        recaptured: bool, whether atom1 atom is recaptured at end of tlist
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
    m_atom1 = cfg.mass_atom1
    m_atom2 = cfg.mass_atom2

    x0_atom1_um, v0_atom1_umps, n1 = sample_QHO_initial_3d(
        m_atom1, np.array(cfg.omega_trap_atom1_Hz) * cfg.load_factor, cfg.T_uK_atom1, rng
    )
    # squeeze position and momentum if needed
    x0_atom1_um = x0_atom1_um / np.array(cfg.squeeze_factor)
    v0_atom1_umps = v0_atom1_umps * np.array(cfg.squeeze_factor)

    x0_atom2_um, v0_atom2_umps, n2 = sample_QHO_initial_3d(
        m_atom2, np.array(cfg.omega_trap_atom2_Hz) * cfg.load_factor, cfg.T_uK_atom2, rng
    )
    # squeeze position and momentum if needed
    x0_atom2_um = x0_atom2_um / np.array(cfg.squeeze_factor)
    v0_atom2_umps = v0_atom2_umps * np.array(cfg.squeeze_factor)
    
    cfg.n_list.append((n1, n2)) # store n values

    cfg.positions1_list.append(x0_atom1_um)
    cfg.velocities1_list.append(v0_atom1_umps) # store displacement over tlist
    cfg.positions2_list.append(x0_atom2_um)
    cfg.velocities2_list.append(v0_atom2_umps) # store displacement over tlist
    # Relative position over time (μm)
    # R(t) = R0 + (x0_atom1 - x0_atom2) + (v_atom1 - v_atom2)*t
    dx = (x0_atom1_um - x0_atom2_um)
    dv = (v0_atom1_umps - v0_atom2_umps)
    R_vec_t = R0_vec_um + dx + dv * (tlist[:, None] + cfg.t_pi_atom2_us * 1e-6) # include motion during atom2 π pulse
    # print(f"Initial relative position dx: {dx} μm")
    R_norm_t = np.linalg.norm(R_vec_t, axis=1)
    positions_list.append(dx)
    velocities_list.append(dv)
    
    V_t = c6_rad_um6 / (R_norm_t ** 6)

    def V_of_t(_times: np.ndarray) -> np.ndarray:
        # QuTiP will pass the same tlist; return precomputed array
        return V_t
    energy_array1 = None
    energy_array2 = None

    if not cfg.simulate_atom1_recapture and not cfg.simulate_atom2_recapture: # neither atom recapture simulated
        recaptured = True
        return V_of_t, recaptured
    if cfg.simulate_atom1_recapture:
        # check if atom1 is recaptured (we are dominated by atom1 loss)
        recaptured1, energy_array = is_atom_recaptured(cfg,
            x0_atom1_um, v0_atom1_umps,
            m_kg=m_atom1,
            omegas_Hz_xyz=cfg.omega_trap_atom1_Hz,
            Temps_uK_xyz=cfg.T_uK_atom1,
            load_depth_T_uK=cfg.load_depth_T_atom1,
            tlist=tlist + cfg.t_pi_atom2_us * 1e-6, # include motion during atom2 π pulse
            wavelength_nm=cfg.wavelength_nm_atom1,
            rng=rng
        ) # True if recaptured, False if lost
        recaptured = recaptured1
        cfg.energy_array1.append(energy_array)
    if cfg.simulate_atom2_recapture:
        # check if atom2 is recaptured (we are dominated by atom2 loss)
        recaptured2, energy_array = is_atom_recaptured(cfg,
            x0_atom2_um, v0_atom2_umps,
            m_kg=m_atom2,
            omegas_Hz_xyz=cfg.omega_trap_atom2_Hz,
            Temps_uK_xyz=cfg.T_uK_atom2,
            load_depth_T_uK=cfg.load_depth_T_atom2,
            tlist=tlist + cfg.t_pi_atom2_us * 1e-6, # include motion during atom2 π pulse
            wavelength_nm=cfg.wavelength_nm_atom2,
            rng=rng
        ) # True if recaptured, False if lost
        recaptured = recaptured2
        cfg.energy_array2.append(energy_array)
    if cfg.simulate_atom1_recapture and cfg.simulate_atom2_recapture:
        recaptured = recaptured1 and recaptured2
    # print(V_t)
    return V_of_t, recaptured

# Single-shot simulation
# ----------------------------
def simulate_shot(Delta_atom1_Hz,
                  cfg,
                  c6_rad_um6,
                  rng,
                  T1_cache=None,
                  interaction_on=True,
                  positions_list=[],
                  velocities_list=[]):
    """
    Simulate one 'shot' of the atom1 π-pulse.
    If interaction_on=False, the V(R) term is set to zero.
    Parameters:
        Delta_atom1_Hz: atom1 detuning for this shot (Hz)
        cfg: ExperimentConfig
        c6_rad_um6: precomputed C6 coefficient
        rng: np.random.Generator
        T1_cache: optional tuple of (T1_atom1, T1_atom2) to avoid recomputing lifetimes
        interaction_on: whether to include interaction term in Hamiltonian
    Returns: P_excitation (float)
    """
    OMEGA_atom1_rad = 2 * np.pi * cfg.OMEGA_atom1_Hz
    OMEGA_atom2_rad = 2 * np.pi * cfg.OMEGA_atom2_Hz
    Delta_atom1_rad = 2 * np.pi * Delta_atom1_Hz

    # π-pulse time and time list
    if cfg.t_pi_atom1_us == None:
        t_pi = np.pi / max(OMEGA_atom1_rad, 1e-30) # avoid div by zero [sec]
    else:
        t_pi = cfg.t_pi_atom1_us * 1e-6  # [sec]

    # time list for evolution contains pi pulse for each atom + wait time
    tlist = np.linspace(0.0, t_pi, cfg.N_steps) # [sec]

    # Prepare interaction function
    if interaction_on:
        V_of_t, recaptured = make_V_of_t_generator(
            c6_rad_um6=c6_rad_um6,
            cfg=cfg,
            tlist=tlist,
            rng=rng,
            positions_list=positions_list,
            velocities_list=velocities_list
        )
        if not recaptured:
            # Atom lost; return 0 excitation probability immediately
            return 0.0
    else:
        # No interaction -> V(t) = 0 for all times
        V_of_t = lambda t: np.zeros_like(t)

    if cfg.hamiltonian_on == False: # skip Hamiltonian evolution just return 0 population
        population = 1.0
        return population

    # if atom is recaptured and we want to simulate Hamiltonian, proceed to build Hamiltonian and simulate

    # Build Hamiltonian
    H = build_time_dependent_H(
        OMEGA_atom1_rad=OMEGA_atom1_rad,
        OMEGA_atom2_rad=OMEGA_atom2_rad,
        Delta_atom1_rad=Delta_atom1_rad,
        V_of_t=V_of_t,
        tlist=tlist,
    )

    # Initial state |g_atom1, r_atom2>
    g = basis(2, 0)
    r = basis(2, 1)
    psi0 = tensor(g, r)

    # Collapses (use cached lifetimes)
    if T1_cache is not None:
        T1_atom1, T1_atom2 = T1_cache
    else:
        T1_atom1, T1_atom2 = 50e-6, 80e-6

    n_atom1, n_atom2, _, _, _, sz_atom1, sz_atom2, sm_atom1, sm_atom2 = two_atom_ops()
    c_ops = []
    if T1_atom1 > 0:
        c_ops.append(np.sqrt(1.0 / T1_atom1) * sm_atom1)
    if T1_atom2 > 0:
        c_ops.append(np.sqrt(1.0 / T1_atom2) * sm_atom2)
    # Dephasing rates
    gamma_phi_atom1 = max(0.0, (1.0 / cfg.T2_atom1_s) - 0.5 * (0.0 if T1_atom1 == 0 else 1.0 / T1_atom1))
    gamma_phi_atom2 = max(0.0, (1.0 / cfg.T2_atom2_s) - 0.5 * (0.0 if T1_atom2 == 0 else 1.0 / T1_atom2))
    if gamma_phi_atom1 > 0:
        c_ops.append(np.sqrt(0.5 * gamma_phi_atom1) * sz_atom1)
    if gamma_phi_atom2 > 0:
        c_ops.append(np.sqrt(0.5 * gamma_phi_atom2) * sz_atom2)
    
    result = mesolve(H, psi0, tlist, c_ops=c_ops, e_ops=[n_atom1])
    population = 1-float(np.real(result.expect[0][-1]))

    return population

# Detuning scan (Monte Carlo)
# ----------------------------
def scan_detuning(cfg, c6_rad_um6=None, interaction_on=True):
    """
    Scan over atom1 detunings with Monte Carlo sampling.
    Parameters:
        cfg: ExperimentConfig
        c6_rad_um6: optional precomputed C6 coefficient
        interaction_on: whether to include interaction term in Hamiltonian
    Returns: (Delta_scan_Hz, P_excitation)
    """
    cfg.energy_array1 = [] # to store energy arrays if recapture simulated
    cfg.energy_array2 = [] # to store energy arrays if recapture simulated
    cfg.positions1_list = [] # to store positions if recapture simulated
    cfg.positions2_list = [] # to store positions if recapture simulated
    cfg.velocities1_list = [] # to store velocities if recapture simulated
    cfg.velocities2_list = [] # to store velocities if recapture simulated
    cfg.n_list = [] # to store n values if recapture simulated

    rng = np.random.default_rng(cfg.seed)
    if c6_rad_um6 is None:
        c6_rad_um6 = compute_c6_atom1_atom2_rad_per_s_um6(cfg)

    if cfg.T1_use_ARC:
        T1_atom1, T1_atom2 = get_state_lifetimes_from_arc(
            cfg,
            # temperature_atom1_K=np.mean(cfg.T_uK_atom1) * 1e-6,
            # temperature_atom2_K=np.mean(cfg.T_uK_atom2) * 1e-6,
            # n_l_j_atom1=(cfg.n_atom1, cfg.l_atom1, cfg.j_atom1),
            # n_l_j_atom2=(cfg.n_atom2, cfg.l_atom2, cfg.j_atom2),
            n_upper_offset=0, # arbitrary cutoff for levels included in ARC lifetime calc
        )

    else:
        T1_atom1, T1_atom2 = 0, 0

    if interaction_on & cfg.broadening:
        cfg.positions_list = []  # to store positions for analysis
        cfg.velocities_list = []  # to store velocities for analysis

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
                dHz, cfg, c6_rad_um6, rng, T1_cache=(T1_atom1, T1_atom2),
                interaction_on=interaction_on,
                positions_list=cfg.positions_list,
                velocities_list=cfg.velocities_list
            )
        P[i] = acc / cfg.N_mc # average over MC shots

    # plot average displacement for debugging
    if cfg.plot_displacements and interaction_on & cfg.broadening:
        # # plot histogram of positions
        # plt.figure(figsize=(8,5))
        # all_displacements = np.concatenate(cfg.displacement_list)
        # plt.hist(all_displacements, bins=50, density=True, alpha=0.7, color='blue')
        # plt.xlabel("Interatomic distance |R| (μm)")
        # plt.ylabel("Probability density")
        # plt.title("Histogram of interatomic distances during interaction")
        # plt.axvline(cfg.R_mean_um, color='red', linestyle='--', label='Mean distance')
        # plt.legend()
        # plt.tight_layout()
        # plt.show()
        # plot dots for positions and velocities
        plt.figure(figsize=(8,5))
        positions_array = np.array(cfg.positions_list)
        velocities_array = np.array(cfg.velocities_list)
        plt.scatter(positions_array[:,0], velocities_array[:,0], alpha=0.5)
        plt.xlabel("Initial relative position dx (μm)")
        plt.ylabel("Initial relative velocity dv (μm/s)")
        plt.title("Initial relative positions and velocities")
        # plt.axvline(0, color='red', linestyle='--', label='Mean position')
        plt.legend()
        plt.tight_layout()
        plt.show()
        
    return cfg.Delta_scan_Hz, P


###############################
### fitting and plotting utils
###############################
def lorentzian(x, A, x0, gamma):
    return 1 - A * (gamma**2 / ((x - x0)**2 + gamma**2))