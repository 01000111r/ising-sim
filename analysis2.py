#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 15:26:16 2025

@author: giancarloramirez
"""
import numpy as np
import matplotlib.pyplot as plt
import glob
import natsort
import os
from scipy.interpolate import UnivariateSpline
from scipy.optimize import brentq

# =============================================================================
# PART 1: Compute and Save Binder Values
# =============================================================================
def compute_and_save_binder_values(input_folder="data6", output_folder="data6", sizes=[8, 16, 32]):
    """
    Computes the Binder cumulant for each system size and saves the values into the output folder.
    If the binder file for a given size already exists, it is used instead of recalculating.
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

        for f in files:
            data = np.load(f)
            try:
                # Extract temperature from filename. Assumes the filename contains 'T<number>N'
                T = float(f.split("T")[1].split("N")[0])
            except Exception as e:
                print(f"Error parsing temperature from {f}: {e}")
                continue

            mag = data['magnetisation']
            m2 = np.mean(mag**2)
            if m2 == 0:
                continue
            binder = 1 - np.mean(mag**4) / (3. * m2**2)
            Ts.append(T)
            bs.append(binder)
  
        # Save sorted Binder values
        sort_idx = np.argsort(Ts)
        np.savez(binder_file, T=np.array(Ts)[sort_idx], binder=np.array(bs)[sort_idx])
        print(f"Saved binder values for N={N} to {binder_file}")


# =============================================================================
# PART 2: Load and Plot Binder Values
# =============================================================================
def load_binder_values(folder="data6", sizes=[8, 16, 32]):
    """
    Loads the saved Binder cumulant values from the specified folder.
    """
    binder_data = {}
    for N in sizes:
        filename = os.path.join(folder, f"binder_N{N}.npz")
        if os.path.exists(filename):
            data = np.load(filename)
            binder_data[N] = {"T": data["T"], "binder": data["binder"]}
        else:
            print(f"Binder file not found for N={N}.")
    return binder_data


def plot_binder_values(binder_data):
    """
    Plots the Binder cumulant vs temperature for all available system sizes.
    """
    plt.figure(figsize=(8, 6))
    for N, data in binder_data.items():
        plt.scatter(data["T"], data["binder"], label=f"N={N}")
    plt.axvline(x=2.269, color='red', linestyle='--', label="Expected $T_c$ (2.269)")
    plt.xlabel("Temperature T")
    plt.ylabel("Binder Cumulant")
    plt.legend()
    plt.grid(True)
    plt.title("Binder Cumulant vs. Temperature")
    plt.show()


# =============================================================================
# PART 3: Critical Temperature (Intersection) Calculation
# =============================================================================
def build_splines(binder_data, s=0):
    """
    Creates a UnivariateSpline interpolation for each system size.
    """
    splines = {}
    for N, data in binder_data.items():
        T = data["T"]
        binder = data["binder"]
        splines[N] = UnivariateSpline(T, binder, s=s)
    return splines


def find_intersection(spline1, spline2, T_min, T_max, num_points=1000):
    """
    Finds the temperature at which two splines intersect.
    """
    T_values = np.linspace(T_min, T_max, num_points)
    diff = spline1(T_values) - spline2(T_values)

    for i in range(len(diff) - 1):
        if diff[i] * diff[i + 1] < 0:
            try:
                return brentq(lambda T: spline1(T) - spline2(T), T_values[i], T_values[i + 1])
            except ValueError:
                continue
    return None


def calculate_all_intersections(binder_data, s=0):
    """
    Computes the intersection temperature for every unique pair of system sizes.
    Returns a dictionary where keys are tuples (N1, N2) and values are the intersection temperatures.
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
            Tc = find_intersection(splines[N1], splines[N2], T_min, T_max)
            if Tc is not None:
                intersection_dict[(N1, N2)] = Tc
                print(f"Intersection for N={N1} and N={N2}: Tc = {Tc:.4f}")
            else:
                print(f"No intersection found for N={N1} and N={N2}.")
    return intersection_dict


def save_intersections(intersections, filename="critical_intersections.npz"):
    """
    Saves the intersection temperatures to a file.
    The keys are stored as strings of the form "N1_N2".
    """
    intersections_to_save = {f"{k[0]}_{k[1]}": v for k, v in intersections.items()}
    np.savez(filename, **intersections_to_save)
    print(f"Saved intersections to {filename}")


def calculate_critical_temperatures(binder_data):
    """
    Computes the critical temperature for each system size using the smallest size as a reference.
    Returns a dictionary of critical temperatures (keys: system sizes, values: T_c) and the reference size.
    """
    splines = build_splines(binder_data)
    reference_size = min(binder_data.keys())
    ref_spline = splines[reference_size]
    critical_temps = {}
    for N, spline in splines.items():
        if N == reference_size:
            continue
        T_min = max(np.min(binder_data[reference_size]["T"]), np.min(binder_data[N]["T"]))
        T_max = min(np.max(binder_data[reference_size]["T"]), np.max(binder_data[N]["T"]))
        Tc = find_intersection(ref_spline, spline, T_min, T_max)
        if Tc is not None:
            critical_temps[N] = Tc
            print(f"Intersection for N={N} (ratio {N/reference_size:.2f}): Tc = {Tc:.4f}")
        else:
            print(f"No intersection found for N={N}.")
    return critical_temps, reference_size


# =============================================================================
# PART 4: Plot Zoomed-Out Graph Showing All Intersections
# =============================================================================
def plot_all_intersections(binder_data, intersections, zoom_margin=0.005, custom_T_low=None, custom_T_high=None):
    """
    Plots a view showing all intersections among the Binder cumulant curves.
    Each intersection point is annotated with its temperature.
    
    You can adjust the zoom either by setting the zoom_margin (which is applied to the min and max 
    of the intersection temperatures) or by directly providing custom_T_low and custom_T_high values.
    """
    splines = build_splines(binder_data)
    
    # Determine zoom window either from provided custom values or automatically from intersections.
    if custom_T_low is not None and custom_T_high is not None:
        T_low, T_high = custom_T_low, custom_T_high
    else:
        if intersections:
            Tc_values = list(intersections.values())
            T_low = min(Tc_values) - zoom_margin
            T_high = max(Tc_values) + zoom_margin
        else:
            T_low, T_high = 2.25, 2.29

    T_fine = np.linspace(T_low, T_high, 1000)
    plt.figure(figsize=(8, 6))
    
    # Plot spline curves and original data points
    for N, spline in splines.items():
        plt.plot(T_fine, spline(T_fine), label=f"N={N}")
        # Plot original data (only within the zoom window)
        data = binder_data[N]
        mask = (data["T"] >= T_low) & (data["T"] <= T_high)
        plt.scatter(np.array(data["T"])[mask], np.array(data["binder"])[mask], s=10)

    # Mark each intersection with a black circle and annotate the temperature.
    for (N1, N2), Tc in intersections.items():
        binder_val = splines[N1](Tc)  # (They should be nearly the same for both splines)
        plt.plot(Tc, binder_val, 'ko', markersize=8)
        plt.text(Tc, binder_val, f" {Tc:.3f}", fontsize=9, color='black')
        plt.axvline(x=Tc, color='gray', linestyle='--', linewidth=0.5)
    
    # Mark the expected critical temperature if within the zoom window.
    if T_low <= 2.269 <= T_high:
        plt.axvline(x=2.269, color='red', linestyle='--', label="Expected $T_c$ (2.269)")
    
    plt.xlabel("Temperature T")
    plt.ylabel("Binder Cumulant")
    plt.title("Zoomed-Out View: Binder Curve Intersections")
    plt.legend()
    plt.grid(True)
    plt.show()


# =============================================================================
# PART 5: Plot Critical Temperature vs. System Size Ratio
# =============================================================================
def plot_critical_temperatures(critical_temps, reference_size):
    """
    Plots the critical temperature as a function of system size ratio.
    """
    ratios = [N / reference_size for N in critical_temps.keys()]
    Tc_values = list(critical_temps.values())

    plt.figure(figsize=(8, 6))
    plt.scatter(ratios, Tc_values, color='blue', label="Estimated $T_c$")
    plt.axhline(y=2.269, color='red', linestyle='--', label="Expected $T_c$ (2.269)")
    plt.xlabel("System Size Ratio ($N / N_{small}$)")
    plt.ylabel("Critical Temperature $T_c$")
    plt.title("Critical Temperature vs. System Size Ratio")
    plt.legend()
    plt.grid(True)
    plt.show()


# =============================================================================
# PART 6: Load and Plot Observables (Energy, Magnetisation, Susceptibility, Specific Heat)
# =============================================================================
def load_observables(input_folder="data6", sizes=[8, 16, 32]):
    """
    Loads observable data from the specified folder for each system size.
    For each file, extracts the temperature and the following observables:
    energy, magnetisation, susceptibility, and specific_heat.
    Returns a dictionary indexed by system size.
    """
    observables_data = {}
    for N in sizes:
        files = natsort.natsorted(glob.glob(f"{input_folder}/*N{N}*.npz"))
        Ts = []
        energy = []
        magnetisation = []
        susceptibility = []
        specific_heat = []
        for f in files:
            data = np.load(f)
            try:
                # Extract temperature from filename (assumes filename contains 'T<number>N')
                T = float(f.split("T")[1].split("N")[0])
            except Exception as e:
                print(f"Error parsing temperature from {f}: {e}")
                continue
            Ts.append(T)
            # Append the mean value of each observable if it exists; else, use NaN.
            energy.append(np.mean(data['energy']) if 'energy' in data else np.nan)
            magnetisation.append(np.mean(data['magnetisation']) if 'magnetisation' in data else np.nan)
            susceptibility.append(np.mean(data['susceptibility']) if 'susceptibility' in data else np.nan)
            specific_heat.append(np.mean(data['specific_heat']) if 'specific_heat' in data else np.nan)
        sort_idx = np.argsort(Ts)
        observables_data[N] = {
            "T": np.array(Ts)[sort_idx],
            "energy": np.array(energy)[sort_idx],
            "magnetisation": np.array(magnetisation)[sort_idx],
            "susceptibility": np.array(susceptibility)[sort_idx],
            "specific_heat": np.array(specific_heat)[sort_idx]
        }
    return observables_data


def plot_observables(observables_data):
    """
    Plots four graphs (energy, magnetisation, susceptibility, specific heat versus temperature)
    in one large figure. Each subplot shows the data for all system sizes.
    """
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # Energy vs Temperature
    ax = axs[0, 0]
    for N, data in observables_data.items():
        ax.scatter(data["T"], data["energy"], marker='o', linestyle='-', label=f"N={N}")
    ax.set_xlabel("Temperature T")
    ax.set_ylabel("Energy")
    ax.set_title("Energy vs Temperature")
    ax.legend()
    ax.grid(True)
    
    # Magnetisation vs Temperature
    ax = axs[0, 1]
    for N, data in observables_data.items():
        ax.scatter(data["T"], data["magnetisation"], marker='o', linestyle='-', label=f"N={N}")
    ax.set_xlabel("Temperature T")
    ax.set_ylabel("Magnetisation")
    ax.set_title("Magnetisation vs Temperature")
    ax.legend()
    ax.grid(True)
    
    # Susceptibility vs Temperature
    ax = axs[1, 0]
    for N, data in observables_data.items():
        ax.scatter(data["T"], data["susceptibility"], marker='o', linestyle='-', label=f"N={N}")
    ax.set_xlabel("Temperature T")
    ax.set_ylabel("Susceptibility")
    ax.set_title("Susceptibility vs Temperature")
    ax.legend()
    ax.grid(True)
    
    # Specific Heat vs Temperature
    ax = axs[1, 1]
    for N, data in observables_data.items():
        ax.scatter(data["T"], data["specific_heat"], marker='o', linestyle='-', label=f"N={N}")
    ax.set_xlabel("Temperature T")
    ax.set_ylabel("Specific Heat")
    ax.set_title("Specific Heat vs Temperature")
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.show()


# =============================================================================
# NEW PART 7: Log-Log Plots of Observables (Translated by Critical Temperature)
# =============================================================================
def plot_loglog_observables(observables_data, intersections, cutoff=1e-3):
    """
    Creates log-log plots for four observables (energy, magnetisation, susceptibility,
    and specific heat). The temperature axis is translated by the average critical 
    temperature (T_c, computed as the mean of the intersection temperatures).

    For observables that scale above T_c (energy, susceptibility, specific heat), x = T - T_c.
    For magnetisation (nonzero below T_c), x = T_c - T.

    For energy and magnetisation, we take the absolute value so that log-log plotting is valid.
    A best-fit line is computed for each system size and its slope is annotated.
    """
    # Compute average T_c from intersections (or use default value)
    Tc_avg = np.mean(list(intersections.values())) if intersections else 2.269

    # Define translation direction for each observable.
    # (For energy, susceptibility, specific heat: "above"; for magnetisation: "below")
    obs_direction = {
        "energy": "above",
        "magnetisation": "below",
        "susceptibility": "above",
        "specific_heat": "above"
    }
    
    # Create a 2x2 subplot grid.
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    axs = axs.flatten()
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    # Loop over each observable.
    for i, key in enumerate(obs_direction):
        ax = axs[i]
        side = obs_direction[key]
        
        # Loop over each system size (sorted for consistency).
        for j, N in enumerate(sorted(observables_data.keys())):
            data = observables_data[N]
            T = data["T"]
            y = data[key]
            
            # For energy and magnetisation, take absolute value so that log is defined.
            if key in ["energy", "magnetisation"]:
                y = np.abs(y)
            
            # Translate the temperature axis:
            # For "above": x = T - T_c; for "below": x = T_c - T.
            x = T - Tc_avg if side == "above" else Tc_avg - T
            
            # Filter out data points with very small x values.
            mask = x > cutoff
            x, y = x[mask], y[mask]
            if len(x) < 2:
                continue  # Not enough points to compute a fit
            
            # Scatter plot for this system size.
            color = colors[j % len(colors)]
            ax.scatter(x, y, color=color, label=f"N={N}")
            
            # Compute best-fit line in log-log space.
            logx, logy = np.log(x), np.log(y)
            slope, intercept = np.polyfit(logx, logy, 1)
            x_fit = np.linspace(np.min(x), np.max(x), 100)
            y_fit = np.exp(intercept) * x_fit**slope
            ax.plot(x_fit, y_fit, '--', color=color, label=f"N={N} fit: {slope:.3f}")
            # Annotate the slope near the median x value.
            ax.text(np.median(x), np.median(y_fit), f"{slope:.3f}", color=color,
                    fontsize=9, bbox=dict(facecolor='white', alpha=0.5))
        
        # Set log scales and labels.
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
# MAIN EXECUTION: Run the Workflow
# =============================================================================
if __name__ == "__main__":
    # Define the folder where the simulation outputs (and simulation_parameters.npz) are saved
    folder = "data8"
    
    # -------------------------------
    # Extract simulation parameters from the NPZ file
    # -------------------------------
    params_filepath = os.path.join(folder, "simulation_parameters.npz")
    if os.path.exists(params_filepath):
        params = np.load(params_filepath)
        nt = int(params["nt"])
        n_list = params["n_list"].tolist()  # Extract system sizes as a list
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
    
    # Now use the extracted parameters in the analysis workflow:
    compute_and_save_binder_values(input_folder=folder, output_folder=folder, sizes=n_list)
    
    binder_data = load_binder_values(folder=folder, sizes=n_list)
    plot_binder_values(binder_data)
    
    intersections = calculate_all_intersections(binder_data)
    save_intersections(intersections, filename="critical_intersections.npz")
    
    plot_all_intersections(binder_data, intersections, zoom_margin=0.005, custom_T_low=2.25, custom_T_high=2.28)
    
    critical_temps, ref_size = calculate_critical_temperatures(binder_data)
    plot_critical_temperatures(critical_temps, ref_size)
    
    observables_data = load_observables(input_folder=folder, sizes=n_list)
    plot_observables(observables_data)
    
    # NEW: Create log-log plots for the observables using the average T_c from the intersections.
    plot_loglog_observables(observables_data, intersections, cutoff=1e-1)
