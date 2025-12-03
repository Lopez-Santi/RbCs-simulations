"""
Use this to run MATLAB scripts from Python and get variables back as NumPy arrays.
Requires MATLAB Engine for Python to be installed and configured.

Currently setup to have MATLAB open and directly put what data path you want to use in the MATLAB script.
This should later be modified to pass arguments to the MATLAB script.
"""

import numpy as np
import matlab.engine
import matplotlib.pyplot as plt

def run_matlab_script_and_get_vars(script_name, var_names, script_path=None):
    """
    Run a MATLAB script and return specified workspace variables as NumPy arrays.

    Parameters
    ----------
    script_name : str
        Name of the MATLAB script (without '.m' extension), or a command string.
    var_names : list of str
        Names of workspace variables to fetch from MATLAB.
    script_path : str, optional
        Directory path to script to change to before running the script.
    Returns
    -------
    dict
        Mapping from variable name to NumPy array (or Python type).
    """
    # Start MATLAB engine
    eng = matlab.engine.start_matlab()

    # add experiment-control code path and its subfolders
    eng.addpath(eng.genpath(r'/Users/santi/Library/CloudStorage/GoogleDrive-santilopez@g.harvard.edu/My Drive/Research/Code/experiment-control'), nargout=0)

    if script_path:
        eng.cd(script_path, nargout=0)

    # print(eng.pwd())  # print current MATLAB directory
    try:
        # Run the script (close all figures after running)
        eng.eval(script_name + "; close all;", nargout=0)

        # Fetch variables from base workspace
        result = {}
        for name in var_names:
            value = eng.workspace[name]  # get variable from MATLAB base workspace

            # Convert typical MATLAB numeric arrays to NumPy
            # (matlab.double, matlab.single, etc. become numpy arrays)
            try:
                arr = np.array(value)
            except TypeError:
                # For non-numeric or exotic types, just keep the raw object
                arr = value

            result[name] = arr

        return result

    finally:
        # Always close the engine
        eng.quit()