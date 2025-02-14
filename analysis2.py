#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 15:26:16 2025
Updated on Feb 13 2025

@author: giancarloramirez
This analysis file loads simulation data, computes error–estimates for observables 
(including Binder cumulant and its intersections), and produces various plots including 
a new one: observable errors vs temperature.
"""

import numpy as np
import matplotlib.pyplot as plt
import glob
import natsort
import os
from scipy.interpolate import UnivariateSpline
from scipy.optimize import brentq

# =============================================================================
# PART 1: Compute and Save Binder Values and Observable Errors
# =============================================================================
def compute_and_save_binder_values(input_folder="data6", output_folder="data6", sizes=[8, 16, 32]):
    """
    For each system size, loops over all simulation files, computes the Binder cumulant
    and its error (using the magnetisation array) and also computes the means and errors 
    for energy, magnetisation, susceptibility, and specific heat. All these are then saved 
    to a single NPZ file per system size.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for N in sizes:
        binder_file = os.path.join(output_folder, f"binder_N{N}.npz")
        if os.path.exists(binder_file):
            print(f"Binder file for N={N} already exists. Skipping computation.")
            continue

        files = natsort.natsorted(glob.glob(f"{input_folder}/*N{N}*.npz"))
        print(f"Processing system size N={N}")
        
        Ts = []
        bs = []
        binder_errors = []
        
        # For observables:
        energy_means = []
        energy_errors = []
        magnetisation_means = []
        magnetisation_errors = []
        susceptibility_vals = []
        susceptibility_errors = []
        specific_heat_vals = []
        specific_heat_errors = []
        
        for f in files:
            data = np.load(f)
            try:
                # Assumes the filename contains 'T<number>N'
                T = float(f.split("T")[1].split("N")[0])
            except Exception as e:
                print(f"Error parsing temperature from {f}: {e}")
                continue

            # ---------------------------
            # Process magnetisation for Binder and M observables
            # ---------------------------
            if 'magnetisation' not in data:
                continue
            mag = data['magnetisation']
            n_mag = len(mag)
            m_mean = np.mean(mag)
            m_std = np.std(mag, ddof=1)
            m_err = m_std / np.sqrt(n_mag)
            
            # Compute Binder cumulant using m^2 and m^4:
            m2 = np.mean(mag**2)
            m4 = np.mean(mag**4)
            std_m2 = np.std(mag**2, ddof=1)
            std_m4 = np.std(mag**4, ddof=1)
            err_m2 = std_m2 / np.sqrt(n_mag)
            err_m4 = std_m4 / np.sqrt(n_mag)
            
            # Avoid division by zero:
            if m2 == 0:
                continue
            binder = 1 - m4 / (3 * m2**2)
            # Propagate error: ∂U/∂⟨m^4⟩ = -1/(3 m2²) and ∂U/∂⟨m²⟩ = 2 m4/(3 m2³)
            binder_err = np.sqrt(((2 * m4 / (3 * m2**3)) * err_m2)**2 +
                                 ((1 / (3 * m2**2)) * err_m4)**2)
            
            # ---------------------------
            # Process energy observable
            # ---------------------------
            if 'energy' in data:
                energy = data['energy']
                n_e = len(energy)
                e_mean = np.mean(energy)
                e_std = np.std(energy, ddof=1)
                e_err = e_std / np.sqrt(n_e)
            else:
                e_mean = np.nan
                e_err = np.nan

            # ---------------------------
            # Process susceptibility
            # Susceptibility = beta * (⟨m^2⟩ - ⟨m⟩^2) with beta=1/T.
            # Propagate error: Δf = sqrt((err_m2)² + (2*m_mean*m_err)²)
            # so Δχ = beta * Δf.
            # ---------------------------
            beta = 1.0 / T
            susceptibility_val = beta * (m2 - m_mean**2)
            error_f = np.sqrt(err_m2**2 + (2 * m_mean * m_err)**2)
            susceptibility_err = beta * error_f

            # ---------------------------
            # Process specific heat
            # Specific heat = beta^2 * (⟨e^2⟩ - ⟨e⟩^2)
            # ---------------------------
            if 'energy' in data:
                e2 = np.mean(energy**2)
                std_e2 = np.std(energy**2, ddof=1)
                e2_err = std_e2 / np.sqrt(n_e)
                specific_heat_val = beta**2 * (e2 - e_mean**2)
                specific_heat_err = beta**2 * np.sqrt(e2_err**2 + (2 * e_mean * e_err)**2)
            else:
                specific_heat_val = np.nan
                specific_heat_err = np.nan

            # ---------------------------
            # Append computed values
            # ---------------------------
            Ts.append(T)
            bs.append(binder)
            binder_errors.append(binder_err)
            
            energy_means.append(e_mean)
            energy_errors.append(e_err)
            magnetisation_means.append(m_mean)
            magnetisation_errors.append(m_err)
            susceptibility_vals.append(susceptibility_val)
            susceptibility_errors.append(susceptibility_err)
            specific_heat_vals.append(specific_heat_val)
            specific_heat_errors.append(specific_heat_err)
        
        # Sort the arrays by temperature.
        sort_idx = np.argsort(Ts)
        np.savez(binder_file,
                 T=np.array(Ts)[sort_idx],
                 binder=np.array(bs)[sort_idx],
                 binder_error=np.array(binder_errors)[sort_idx],
                 energy_mean=np.array(energy_means)[sort_idx],
                 energy_error=np.array(energy_errors)[sort_idx],
                 magnetisation_mean=np.array(magnetisation_means)[sort_idx],
                 magnetisation_error=np.array(magnetisation_errors)[sort_idx],
                 susceptibility=np.array(susceptibility_vals)[sort_idx],
                 susceptibility_error=np.array(susceptibility_errors)[sort_idx],
                 specific_heat=np.array(specific_heat_vals)[sort_idx],
                 specific_heat_error=np.array(specific_heat_errors)[sort_idx])
        print(f"Saved binder and observables for N={N} to {binder_file}")

# =============================================================================
# PART 2: Load and Plot Binder Values (with error bars)
# =============================================================================
def load_binder_values(folder="data6", sizes=[8, 16, 32]):
    """
    Loads the saved Binder cumulant values (and associated observable data) from the folder.
    """
    binder_data = {}
    for N in sizes:
        filename = os.path.join(folder, f"binder_N{N}.npz")
        if os.path.exists(filename):
            data = np.load(filename)
            binder_data[N] = {"T": data["T"],
                              "binder": data["binder"],
                              "binder_error": data["binder_error"],
                              "energy_mean": data["energy_mean"],
                              "energy_error": data["energy_error"],
                              "magnetisation_mean": data["magnetisation_mean"],
                              "magnetisation_error": data["magnetisation_error"],
                              "susceptibility": data["susceptibility"],
                              "susceptibility_error": data["susceptibility_error"],
                              "specific_heat": data["specific_heat"],
                              "specific_heat_error": data["specific_heat_error"]}
        else:
            print(f"Binder file not found for N={N}.")
    return binder_data

def plot_binder_values(binder_data):
    """
    Plots the Binder cumulant vs. temperature for all system sizes with error bars.
    """
    plt.figure(figsize=(8, 6))
    for N, data in binder_data.items():
        plt.errorbar(data["T"], data["binder"], yerr=data["binder_error"],
                     fmt='o', label=f"N={N}", ecolor='black', capsize=5, alpha=0.5)
    plt.axvline(x=2.269, color='red', linestyle='--', label="Expected $T_c$ (2.269)")
    plt.xlabel("Temperature T")
    plt.ylabel("Binder Cumulant")
    plt.legend()
    plt.grid(True)
    plt.title("Binder Cumulant vs. Temperature")
    plt.show()

# =============================================================================
# PART 3: Critical Temperature (Intersection) Calculation with Errors
# =============================================================================
def build_splines(binder_data, s=0):
    """
    Creates a UnivariateSpline interpolation for each system size for both the Binder cumulant
    and its error.
    Returns a dict of the form:
      { N: {"spline": spline, "error_spline": error_spline } }.
    """
    splines = {}
    for N, data in binder_data.items():
        T = data["T"]
        binder = data["binder"]
        spline = UnivariateSpline(T, binder, s=s)
        binder_error = data["binder_error"]
        error_spline = UnivariateSpline(T, binder_error, s=s)
        splines[N] = {"spline": spline, "error_spline": error_spline}
    return splines

def find_intersection(spline1, spline2, error_spline1, error_spline2, T_min, T_max, num_points=1000):
    """
    Finds the temperature at which two binder splines intersect and estimates the error.
    Returns a tuple (Tc, Tc_error) if found, else (None, None).

    Error on Tc is estimated by:
      ΔTc ≈ sqrt((ΔU₁)² + (ΔU₂)²) / |d/dT (U₁ - U₂)| at Tc.
    """
    T_values = np.linspace(T_min, T_max, num_points)
    diff = spline1(T_values) - spline2(T_values)

    for i in range(len(diff) - 1):
        if diff[i] * diff[i+1] < 0:
            try:
                Tc = brentq(lambda T: spline1(T) - spline2(T), T_values[i], T_values[i+1])
                d_diff_dT = spline1.derivative()(Tc) - spline2.derivative()(Tc)
                err1 = error_spline1(Tc)
                err2 = error_spline2(Tc)
                Tc_error = np.sqrt(err1**2 + err2**2) / np.abs(d_diff_dT) if d_diff_dT != 0 else np.inf
                return Tc, Tc_error
            except ValueError:
                continue
    return None, None

def calculate_all_intersections(binder_data, s=0):
    """
    Computes the intersection temperature (and error) for every unique pair of system sizes.
    Returns a dictionary where keys are tuples (N1, N2) and values are tuples (Tc, Tc_error).
    """
    splines = build_splines(binder_data, s=s)
    intersection_dict = {}
    sizes = sorted(binder_data.keys())
    for i in range(len(sizes)):
        for j in range(i+1, len(sizes)):
            N1 = sizes[i]
            N2 = sizes[j]
            T_min = max(np.min(binder_data[N1]["T"]), np.min(binder_data[N2]["T"]))
            T_max = min(np.max(binder_data[N1]["T"]), np.max(binder_data[N2]["T"]))
            Tc, Tc_error = find_intersection(splines[N1]["spline"], splines[N2]["spline"],
                                             splines[N1]["error_spline"], splines[N2]["error_spline"],
                                             T_min, T_max)
            if Tc is not None:
                intersection_dict[(N1, N2)] = (Tc, Tc_error)
                print(f"Intersection for N={N1} and N={N2}: Tc = {Tc:.4f} ± {Tc_error:.4f}")
            else:
                print(f"No intersection found for N={N1} and N={N2}.")
    return intersection_dict

def save_intersections(intersections, filename="critical_intersections.npz"):
    """
    Saves the intersection temperatures and their errors to a file.
    The keys are stored as strings in the form "N1_N2".
    """
    intersections_to_save = {f"{k[0]}_{k[1]}": np.array(v) for k, v in intersections.items()}
    np.savez(filename, **intersections_to_save)
    print(f"Saved intersections to {filename}")

def calculate_critical_temperatures(binder_data):
    """
    Computes the critical temperature for each system size using the smallest size as a reference.
    Returns a dictionary of critical temperatures (keys: system sizes, values: (Tc, Tc_error)) and 
    the reference size.
    """
    splines = build_splines(binder_data)
    reference_size = min(binder_data.keys())
    ref_spline = splines[reference_size]["spline"]
    ref_error_spline = splines[reference_size]["error_spline"]
    critical_temps = {}
    for N, spline_data in splines.items():
        if N == reference_size:
            continue
        T_min = max(np.min(binder_data[reference_size]["T"]), np.min(binder_data[N]["T"]))
        T_max = min(np.max(binder_data[reference_size]["T"]), np.max(binder_data[N]["T"]))
        Tc, Tc_error = find_intersection(ref_spline, spline_data["spline"],
                                         ref_error_spline, spline_data["error_spline"],
                                         T_min, T_max)
        if Tc is not None:
            critical_temps[N] = (Tc, Tc_error)
            print(f"Intersection for N={N} (ratio {N/reference_size:.2f}): Tc = {Tc:.4f} ± {Tc_error:.4f}")
        else:
            print(f"No intersection found for N={N}.")
    return critical_temps, reference_size

# =============================================================================
# PART 4: Plot Zoomed-Out Graph Showing All Intersections (with error bars)
# =============================================================================
def plot_all_intersections(binder_data, intersections, zoom_margin=0.005, custom_T_low=None, custom_T_high=None):
    """
    Plots a view showing all intersections among the Binder cumulant curves.
    Each intersection point is annotated with its temperature and error.
    """
    splines = build_splines(binder_data)
    
    if custom_T_low is not None and custom_T_high is not None:
        T_low, T_high = custom_T_low, custom_T_high
    else:
        if intersections:
            Tc_values = [val[0] for val in intersections.values()]
            T_low = min(Tc_values) - zoom_margin
            T_high = max(Tc_values) + zoom_margin
        else:
            T_low, T_high = 2.25, 2.29

    T_fine = np.linspace(T_low, T_high, 1000)
    plt.figure(figsize=(8, 6))
    
    for N, spline_data in splines.items():
        spline = spline_data["spline"]
        plt.plot(T_fine, spline(T_fine), label=f"N={N}")
        data = binder_data[N]
        mask = (data["T"] >= T_low) & (data["T"] <= T_high)
        plt.errorbar(np.array(data["T"])[mask], np.array(data["binder"])[mask],
                     yerr=np.array(data["binder_error"])[mask], fmt='o', markersize=4,
                     ecolor='black', capsize=5, alpha=0.5)
    
    for (N1, N2), (Tc, Tc_error) in intersections.items():
        binder_val = splines[N1]["spline"](Tc)
        plt.errorbar(Tc, binder_val, yerr=Tc_error, fmt='ko', markersize=8,
                     ecolor='black', capsize=5, alpha=0.5)
        plt.text(Tc, binder_val, f" {Tc:.3f}±{Tc_error:.3f}", fontsize=9, color='black')
        plt.axvline(x=Tc, color='gray', linestyle='--', linewidth=0.5)
    
    if T_low <= 2.269 <= T_high:
        plt.axvline(x=2.269, color='red', linestyle='--', label="Expected $T_c$ (2.269)")
    
    plt.xlabel("Temperature T")
    plt.ylabel("Binder Cumulant")
    plt.title("Zoomed-Out View: Binder Curve Intersections")
    plt.legend()
    plt.grid(True)
    plt.show()

# =============================================================================
# PART 5: Plot Critical Temperature vs. System Size (with error bars)
# =============================================================================
def plot_critical_temperatures(critical_temps, reference_size):
    """
    Plots the critical temperature as a function of system size ratio with error bars.
    """
    ratios = [N / reference_size for N in critical_temps.keys()]
    Tc_values = [val[0] for val in critical_temps.values()]
    Tc_errors = [val[1] for val in critical_temps.values()]

    plt.figure(figsize=(8, 6))
    plt.errorbar(ratios, Tc_values, yerr=Tc_errors, fmt='o', color='blue', label="Estimated $T_c$",
                 ecolor='black', capsize=5, alpha=0.5)
    plt.axhline(y=2.269, color='red', linestyle='--', label="Expected $T_c$ (2.269)")
    plt.xlabel("System Size Ratio ($N / N_{small}$)")
    plt.ylabel("Critical Temperature $T_c$")
    plt.title("Critical Temperature vs. System Size")
    plt.legend()
    plt.grid(True)
    plt.show()

# =============================================================================
# PART 6: Load and Plot Observables (Energy, Magnetisation, Susceptibility, Specific Heat)
# =============================================================================
def load_observables(input_folder="data6", sizes=[8, 16, 32]):
    """
    Loads observable data from the binder files for each system size.
    Returns a dictionary indexed by system size.
    """
    observables_data = {}
    for N in sizes:
        binder_file = os.path.join(input_folder, f"binder_N{N}.npz")
        if os.path.exists(binder_file):
            data = np.load(binder_file)
            observables_data[N] = {
                "T": data["T"],
                "energy": data["energy_mean"],
                "energy_error": data["energy_error"],
                "magnetisation": data["magnetisation_mean"],
                "magnetisation_error": data["magnetisation_error"],
                "susceptibility": data["susceptibility"],
                "susceptibility_error": data["susceptibility_error"],
                "specific_heat": data["specific_heat"],
                "specific_heat_error": data["specific_heat_error"]
            }
        else:
            print(f"Binder file not found for N={N}.")
    return observables_data

def plot_observables(observables_data):
    """
    Plots scatter graphs (with error bars) of energy, magnetisation, susceptibility,
    and specific heat versus temperature for all system sizes.
    """
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # Energy vs Temperature
    ax = axs[0, 0]
    for N, data in observables_data.items():
        ax.errorbar(data["T"], data["energy"], yerr=data["energy_error"],
                    fmt='o', label=f"N={N}", ecolor='black', capsize=5, alpha=0.5)
    ax.set_xlabel("Temperature T")
    ax.set_ylabel("Energy")
    ax.set_title("Energy vs Temperature")
    ax.legend()
    ax.grid(True)
    
    # Magnetisation vs Temperature
    ax = axs[0, 1]
    for N, data in observables_data.items():
        ax.errorbar(data["T"], data["magnetisation"], yerr=data["magnetisation_error"],
                    fmt='o', label=f"N={N}", ecolor='black', capsize=5, alpha=0.5)
    ax.set_xlabel("Temperature T")
    ax.set_ylabel("Magnetisation")
    ax.set_title("Magnetisation vs Temperature")
    ax.legend()
    ax.grid(True)
    
    # Susceptibility vs Temperature
    ax = axs[1, 0]
    for N, data in observables_data.items():
        ax.errorbar(data["T"], data["susceptibility"], yerr=data["susceptibility_error"],
                    fmt='o', label=f"N={N}", ecolor='black', capsize=5, alpha=0.5)
    ax.set_xlabel("Temperature T")
    ax.set_ylabel("Susceptibility")
    ax.set_title("Susceptibility vs Temperature")
    ax.legend()
    ax.grid(True)
    
    # Specific Heat vs Temperature
    ax = axs[1, 1]
    for N, data in observables_data.items():
        ax.errorbar(data["T"], data["specific_heat"], yerr=data["specific_heat_error"],
                    fmt='o', label=f"N={N}", ecolor='black', capsize=5, alpha=0.5)
    ax.set_xlabel("Temperature T")
    ax.set_ylabel("Specific Heat")
    ax.set_title("Specific Heat vs Temperature")
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.show()

# =============================================================================
# PART 7: Log-Log Plots of Observables with Error on the Fit Gradient
# =============================================================================
def plot_loglog_observables(observables_data, intersections, cutoff=1e-3):
    """
    Creates log-log plots for energy, magnetisation, susceptibility, and specific heat. 
    The temperature axis is translated by the average critical temperature (T_c).
    Also computes the slope of the best-fit line in log-log space with its error.
    """
    if intersections:
        Tc_avg = np.mean([val[0] for val in intersections.values()])
    else:
        Tc_avg = 2.269

    obs_direction = {
        "energy": "above",
        "magnetisation": "below",
        "susceptibility": "above",
        "specific_heat": "above"
    }
    
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    axs = axs.flatten()
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    for i, key in enumerate(obs_direction):
        ax = axs[i]
        side = obs_direction[key]
        
        for j, N in enumerate(sorted(observables_data.keys())):
            data = observables_data[N]
            T = data["T"]
            y = data[key]
            if key in ["energy", "magnetisation"]:
                y = np.abs(y)
            
            x = T - Tc_avg if side == "above" else Tc_avg - T
            mask = x > cutoff
            x, y = x[mask], y[mask]
            if len(x) < 2:
                continue
            
            color = colors[j % len(colors)]
            ax.errorbar(x, y, fmt='o', color=color, label=f"N={N}",
                        ecolor='black', capsize=5, alpha=0.5)
            
            logx, logy = np.log(x), np.log(y)
            p, cov = np.polyfit(logx, logy, 1, cov=True)
            slope, intercept = p
            slope_error = np.sqrt(cov[0, 0])
            x_fit = np.linspace(np.min(x), np.max(x), 100)
            y_fit = np.exp(intercept) * x_fit**slope
            ax.plot(x_fit, y_fit, '--', color=color,
                    label=f"N={N} fit: {slope:.3f}±{slope_error:.3f}")
            ax.text(np.median(x), np.median(y_fit), f"{slope:.3f}±{slope_error:.3f}",
                    color=color, fontsize=9, bbox=dict(facecolor='white', alpha=0.5))
        
        ax.set_xscale('log')
        ax.set_yscale('log')
        xlabel = r"$T-T_c$" if side == "above" else r"$T_c-T$"
        ax.set_xlabel(xlabel)
        ax.set_ylabel(key.capitalize())
        ax.set_title(f"Log-Log Plot of {key.capitalize()}")
        ax.legend(fontsize='small')
        ax.grid(True, which='both', ls='--')
    
    plt.suptitle(f"Log-Log Plots Translated by $T_c$ (Average $T_c$ = {Tc_avg:.3f})", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# =============================================================================
# NEW PART: Plot Observable Errors vs Temperature
# =============================================================================
def plot_observable_errors(observables_data):
    """
    Plots the errors of the observables vs temperature.
    Creates a 2x2 figure with subplots for:
      - Energy error vs. Temperature
      - Magnetisation error vs. Temperature
      - Susceptibility error vs. Temperature
      - Specific Heat error vs. Temperature
    Each subplot shows data for all system sizes.
    """
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # Energy Error vs Temperature
    ax = axs[0, 0]
    for N, data in observables_data.items():
        ax.scatter(data["T"], data["energy_error"], label=f"N={N}")
    ax.set_xlabel("Temperature T")
    ax.set_ylabel("Energy Error")
    ax.set_title("Energy Error vs. Temperature")
    ax.legend()
    ax.grid(True)
    
    # Magnetisation Error vs Temperature
    ax = axs[0, 1]
    for N, data in observables_data.items():
        ax.scatter(data["T"], data["magnetisation_error"], label=f"N={N}")
    ax.set_xlabel("Temperature T")
    ax.set_ylabel("Magnetisation Error")
    ax.set_title("Magnetisation Error vs. Temperature")
    ax.legend()
    ax.grid(True)
    
    # Susceptibility Error vs Temperature
    ax = axs[1, 0]
    for N, data in observables_data.items():
        ax.scatter(data["T"], data["susceptibility_error"], label=f"N={N}")
    ax.set_xlabel("Temperature T")
    ax.set_ylabel("Susceptibility Error")
    ax.set_title("Susceptibility Error vs. Temperature")
    ax.legend()
    ax.grid(True)
    
    # Specific Heat Error vs Temperature
    ax = axs[1, 1]
    for N, data in observables_data.items():
        ax.scatter(data["T"], data["specific_heat_error"], label=f"N={N}")
    ax.set_xlabel("Temperature T")
    ax.set_ylabel("Specific Heat Error")
    ax.set_title("Specific Heat Error vs. Temperature")
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.show()

# =============================================================================
# MAIN EXECUTION: Run the Analysis Workflow
# =============================================================================
if __name__ == "__main__":
    # Define folder (adjust as needed)
    folder = "data8"
    
    # -------------------------------
    # Load simulation parameters from file if available
    # -------------------------------
    params_filepath = os.path.join(folder, "simulation_parameters.npz")
    if os.path.exists(params_filepath):
        params = np.load(params_filepath)
        nt = int(params["nt"])
        n_list = params["n_list"].tolist()  # List of system sizes
        eqSteps = int(params["eqSteps"])
        mcSteps = int(params["mcSteps"])
        T_arr = params["T_arr"]
        print("Loaded Simulation Parameters:")
        print("nt       =", nt)
        print("n_list   =", n_list)
        print("eqSteps  =", eqSteps)
        print("mcSteps  =", mcSteps)
        print("T_arr    =", T_arr)
    else:
        print(f"Simulation parameters file not found in {folder}. Using default sizes.")
        n_list = [8, 16, 32]
    
    # Compute and save Binder cumulant and observable errors
    compute_and_save_binder_values(input_folder=folder, output_folder=folder, sizes=n_list)
    
    # Load Binder data and plot with error bars
    binder_data = load_binder_values(folder=folder, sizes=n_list)
    plot_binder_values(binder_data)
    
    # Calculate intersections between Binder curves (with error estimation)
    intersections = calculate_all_intersections(binder_data)
    save_intersections(intersections, filename="critical_intersections.npz")
    
    plot_all_intersections(binder_data, intersections, zoom_margin=0.005, custom_T_low=2.25, custom_T_high=2.27)
    
    # Calculate critical temperatures vs. system size (with errors)
    critical_temps, ref_size = calculate_critical_temperatures(binder_data)
    plot_critical_temperatures(critical_temps, ref_size)
    
    # Load and plot other observables with error bars (scatter graph)
    observables_data = load_observables(input_folder=folder, sizes=n_list)
    plot_observables(observables_data)
    
    # Log-Log plots of observables with best-fit slopes (including gradient error)
    plot_loglog_observables(observables_data, intersections, cutoff=1e-1)
    
    # NEW: Plot Observable Errors vs Temperature
    plot_observable_errors(observables_data)
