'''
Function that returns the polarizability for Rb or Cs at a specific wavelength.
Uses the data files from https://www1.udel.edu/atom/

Santi

23 January 2026


'''

import numpy as np
import pandas as pd


def get_polarizability(wavelength_nm, atom="Rb"):
    """Get the polarizability of Rb or Cs at a specific wavelength.
       Rounding the wavelength to 2 decimal places to match data file precision.
    Args:
        wavelength_nm (float): Wavelength in nanometers.
        atom (str): "Rb" or "Cs".
    Returns:
        float: Polarizability in atomic units (a.u.).
    """

    if atom == "Rb":
        df = pd.read_csv("Rb1Pol/Rb1_5s.csv", skiprows=1)
    elif atom == "Cs":
        df = pd.read_csv("Cs1Pol/Cs1_6s.csv", skiprows=1)
    else:
        raise ValueError("Atom must be 'Rb' or 'Cs'")

    df.columns = ["wavelength_nm", "alpha_0", "alpha0_unc", "alpha_2", "alpha2_unc"]
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["wavelength_nm", "alpha_0"])
    alpha = df["alpha_0"].to_numpy()

    # interpolate to find the polarizability at the given wavelength
    polarizability = np.interp(round(wavelength_nm, 2), df["wavelength_nm"], alpha)
    return polarizability