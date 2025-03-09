#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import glob
import natsort
import os
from scipy.interpolate import UnivariateSpline
from scipy.optimize import brentq

def compute_analysis(input_folder, output_folder, sizes):
    """
    

    Parameters
    ----------
    input_folder : TYPE string
        DESCRIPTION. string of input folder name in repo directory
    output_folder : TYPE string
        DESCRIPTION. string of input folder name in repo directory
    sizes : TYPE list
        DESCRIPTION. list of system sizes in input folder

    Returns
    -------
    None.

    """
    #create new folder if dosnt exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    for N in sizes:
        #open file/ create new one
        analysis_file = os.path.join(output_folder, f"Analysis_N{N}.npz")
        #skip if file exits
        if os.path.exists(analysis_file):
            print(f"Analysis file for N={N} already exists. Skipping computation.")
            continue
        #collect and sort runs associated with system size N
        files = natsort.natsorted(glob.glob(f"{input_folder}/*N{N}*.npz"))
        
        print(f"Processing system size N={N}")
        
        #initialising main arrays
        t_vals = []
        binder_vals = []
        binder_errs = []
        e_means = []
        e_errs = []
        mag_means = []
        mag_errs = []
        sus_vals = []
        sus_errs = []
        s_heat_vals = []
        s_heat_errs = []
        
        for f in files:
            
            data = np.load(f)
            
            try:
                #extract temp value from filename (cheapskate method)
                T = float(f.split("T")[1].split("N")[0])
            except Exception as e:
                print(f"Error parsing temperature from {f}: {e}")
                continue
            # Magnetisation calculations
            mag = data['magnetisation']
    
            n_mag = len(mag)
            m_mean = np.mean(mag)
            m_std = np.std(mag, ddof=1)
            m_err = m_std / np.sqrt(n_mag)
            
            m2 = np.mean(mag**2)
            m4 = np.mean(mag**4)
            
            std_m2 = np.std(mag**2, ddof=1)
            std_m4 = np.std(mag**4, ddof=1)
            
            err_m2 = std_m2 / np.sqrt(n_mag)
            err_m4 = std_m4 / np.sqrt(n_mag)
            
            if m2 == 0:
                continue
            # Binder calculations
            binder = 1 - m4 / (3 * m2**2)
            binder_err = np.sqrt(((2 * m4 / (3 * m2**3)) * err_m2)**2 + ((1 / (3 * m2**2)) * err_m4)**2)
            
            #Energy calculations
            e = data['energy']
            n_e = len(e)
            e_mean = np.mean(e)
            e_std = np.std(e, ddof=1)
            e_err = e_std / np.sqrt(n_e)
            
            #Susceptibility and specific heat calculations
            if T == 0:
                continue
            beta = 1.0 / T
            
            sus_val = beta * (m2 - m_mean**2)
            error_f = np.sqrt(err_m2**2 + (2 * m_mean * m_err)**2)
            sus_err = beta * error_f
            
            e2 = np.mean(e**2)
            std_e2 = np.std(e**2, ddof=1)
            
            e2_err = std_e2 / np.sqrt(n_e)
            s_heat_val = beta**2 * (e2 - e_mean**2)
            s_heat_err = beta**2 * np.sqrt(e2_err**2 + (2 * e_mean * e_err)**2)

            #append all values to main lists
            t_vals.append(T)
            binder_vals.append(binder)
            binder_errs.append(binder_err)
            e_means.append(e_mean)
            e_errs.append(e_err)
            mag_means.append(m_mean)
            mag_errs.append(m_err)
            sus_vals.append(sus_val)
            sus_errs.append(sus_err)
            s_heat_vals.append(s_heat_val)
            s_heat_errs.append(s_heat_err)
            
        #sorting index
        sort_idx = np.argsort(t_vals)
        
        #save arrays to associated system size N file names 
        np.savez(analysis_file,
                 t_vals=np.array(t_vals)[sort_idx],
                 binder_vals=np.array(binder_vals)[sort_idx],
                 binder_errs=np.array(binder_errs)[sort_idx],
                 e_means=np.array(e_means)[sort_idx],
                 e_errs=np.array(e_errs)[sort_idx],
                 mag_means=np.array(mag_means)[sort_idx],
                 mag_errs=np.array(mag_errs)[sort_idx],
                 sus_vals=np.array(sus_vals)[sort_idx],
                 sus_errs=np.array(sus_errs)[sort_idx],
                 s_heat_vals=np.array(s_heat_vals)[sort_idx],
                 s_heat_errs=np.array(s_heat_errs)[sort_idx])
        
        print(f"Saved analysis for N={N} to {analysis_file}")

def load_analysis(folder="data6", sizes=[8, 16, 32]):
    analysis_data = {}
    for N in sizes:
        filename = os.path.join(folder, f"Analysis_N{N}.npz")
        if os.path.exists(filename):
            data = np.load(filename)
            analysis_data[N] = {"t_vals": data["t_vals"],
                              "binder_vals": data["binder_vals"],
                              "binder_errs": data["binder_errs"],
                              "e_means": data["e_means"],
                              "e_errs": data["e_errs"],
                              "mag_means": data["mag_means"],
                              "mag_errs": data["mag_errs"],
                              "sus_vals": data["sus_vals"],
                              "sus_errs": data["sus_errs"],
                              "s_heat_vals": data["s_heat_vals"],
                              "s_heat_errs": data["s_heat_errs"]}
        else:
            print(f"Binder file not found for N={N}.")
    return analysis_data

def plot_binder_values(analysis_data):
    plt.figure(figsize=(8, 6))
    for N, data in analysis_data.items():
        plt.errorbar(data["t_vals"], data["binder_vals"], yerr=data["binder_errs"],
                     fmt='o', label=f"N={N}", ecolor='black', capsize=5, alpha=0.5)
    plt.axvline(x=2.269, color='red', linestyle='--', label="Expected $T_c$ (2.269)")
    plt.xlabel("Temperature T")
    plt.ylabel("Binder Cumulant")
    plt.legend()
    plt.grid(True)
    plt.title("Binder Cumulant vs. Temperature")
    plt.show()

def build_splines(analysis_data, s=0):
    splines = {}
    for N, data in analysis_data.items():
        T = data["t_vals"]
        binder = data["binder_vals"]
        spline = UnivariateSpline(T, binder, s=s)
        binder_error = data["binder_errs"]
        error_spline = UnivariateSpline(T, binder_error, s=s)
        splines[N] = {"spline": spline, "error_spline": error_spline}
    return splines

def find_intersection(spline1, spline2, error_spline1, error_spline2, T_min, T_max, num_points=1000):
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

def calculate_all_intersections(analysis_data, s=0):
    splines = build_splines(analysis_data, s=s)
    intersection_dict = {}
    sizes = sorted(analysis_data.keys())
    for i in range(len(sizes)):
        for j in range(i+1, len(sizes)):
            N1 = sizes[i]
            N2 = sizes[j]
            T_min = max(np.min(analysis_data[N1]["t_vals"]), np.min(analysis_data[N2]["t_vals"]))
            T_max = min(np.max(analysis_data[N1]["t_vals"]), np.max(analysis_data[N2]["t_vals"]))
            Tc, Tc_error = find_intersection(splines[N1]["spline"], splines[N2]["spline"],
                                             splines[N1]["error_spline"], splines[N2]["error_spline"],
                                             T_min, T_max)
            if Tc is not None:
                intersection_dict[(N1, N2)] = (Tc, Tc_error)
                print(f"Intersection for N={N1} and N={N2}: Tc = {Tc:.4f} ± {Tc_error:.4f}")
            else:
                print(f"No intersection found for N={N1} and N={N2}.")
    return intersection_dict

# def save_intersections(intersections, filename="critical_intersections.npz"):
#     intersections_to_save = {f"{k[0]}_{k[1]}": np.array(v) for k, v in intersections.items()}
#     np.savez(filename, **intersections_to_save)
#     print(f"Saved intersections to {filename}")

def calculate_critical_temperatures(analysis_data):
    splines = build_splines(analysis_data)
    reference_size = min(analysis_data.keys())
    ref_spline = splines[reference_size]["spline"]
    ref_error_spline = splines[reference_size]["error_spline"]
    critical_temps = {}
    for N, spline_data in splines.items():
        if N == reference_size:
            continue
        T_min = max(np.min(analysis_data[reference_size]["t_vals"]), np.min(analysis_data[N]["t_vals"]))
        T_max = min(np.max(analysis_data[reference_size]["t_vals"]), np.max(analysis_data[N]["t_vals"]))
        Tc, Tc_error = find_intersection(ref_spline, spline_data["spline"],
                                         ref_error_spline, spline_data["error_spline"],
                                         T_min, T_max)
        if Tc is not None:
            critical_temps[N] = (Tc, Tc_error)
            print(f"Intersection for N={N} (ratio {N/reference_size:.2f}): Tc = {Tc:.4f} ± {Tc_error:.4f}")
        else:
            print(f"No intersection found for N={N}.")
    return critical_temps, reference_size

def plot_all_intersections(analysis_data, intersections, zoom_margin=0.005, custom_T_low=None, custom_T_high=None):
    splines = build_splines(analysis_data)
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
        data = analysis_data[N]
        mask = (data["t_vals"] >= T_low) & (data["t_vals"] <= T_high)
        plt.errorbar(np.array(data["t_vals"])[mask], np.array(data["binder_vals"])[mask],
                     yerr=np.array(data["binder_errs"])[mask], fmt='o', markersize=4,
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

def plot_critical_temperatures(critical_temps, reference_size):
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

def load_observables(input_folder="data6", sizes=[8, 16, 32]):
    observables_data = {}
    for N in sizes:
        binder_file = os.path.join(input_folder, f"Analysis_N{N}.npz")
        if os.path.exists(binder_file):
            data = np.load(binder_file)
            observables_data[N] = {
                "t_vals": data["t_vals"],
                "e_means": data["e_means"],
                "e_errs": data["e_errs"],
                "mag_means": data["mag_means"],
                "mag_errs": data["mag_errs"],
                "sus_vals": data["sus_vals"],
                "sus_errs": data["sus_errs"],
                "s_heat_vals": data["s_heat_vals"],
                "s_heat_errs": data["s_heat_errs"]
            }
        else:
            print(f"Binder file not found for N={N}.")
    return observables_data

def plot_observables(observables_data):
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    ax = axs[0, 0]
    for N, data in observables_data.items():
        ax.errorbar(data["t_vals"], data["e_means"], yerr=data["e_errs"],
                    fmt='o', label=f"N={N}", ecolor='black', capsize=5, alpha=0.5)
    ax.set_xlabel("Temperature T")
    ax.set_ylabel("Energy")
    ax.set_title("Energy vs Temperature")
    ax.legend()
    ax.grid(True)
    ax = axs[0, 1]
    for N, data in observables_data.items():
        ax.errorbar(data["t_vals"], data["mag_means"], yerr=data["mag_errs"],
                    fmt='o', label=f"N={N}", ecolor='black', capsize=5, alpha=0.5)
    ax.set_xlabel("Temperature T")
    ax.set_ylabel("Magnetisation")
    ax.set_title("Magnetisation vs Temperature")
    ax.legend()
    ax.grid(True)
    ax = axs[1, 0]
    for N, data in observables_data.items():
        ax.errorbar(data["t_vals"], data["sus_vals"], yerr=data["sus_errs"],
                    fmt='o', label=f"N={N}", ecolor='black', capsize=5, alpha=0.5)
    ax.set_xlabel("Temperature T")
    ax.set_ylabel("Susceptibility")
    ax.set_title("Susceptibility vs Temperature")
    ax.legend()
    ax.grid(True)
    ax = axs[1, 1]
    for N, data in observables_data.items():
        ax.errorbar(data["t_vals"], data["s_heat_vals"], yerr=data["s_heat_errs"],
                    fmt='o', label=f"N={N}", ecolor='black', capsize=5, alpha=0.5)
    ax.set_xlabel("Temperature T")
    ax.set_ylabel("Specific Heat")
    ax.set_title("Specific Heat vs Temperature")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()

def plot_loglog_observables(observables_data, intersections, cutoff=1e-3):
    if intersections:
        Tc_avg = np.mean([val[0] for val in intersections.values()])
    else:
        Tc_avg = 2.269
    obs_keys = {"energy": "e_means",
                "magnetisation": "mag_means",
                "susceptibility": "sus_vals",
                "specific_heat": "s_heat_vals"}
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    axs = axs.flatten()
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i, obs in enumerate(obs_keys.keys()):
        ax = axs[i]
        side = "above" if obs in ["energy", "susceptibility", "specific_heat"] else "below"
        for j, N in enumerate(sorted(observables_data.keys())):
            data = observables_data[N]
            T = data["t_vals"]
            y = data[obs_keys[obs]]
            if obs in ["energy", "magnetisation"]:
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
        ax.set_ylabel(obs.capitalize())
        ax.set_title(f"Log-Log Plot of {obs.capitalize()}")
        ax.legend(fontsize='small')
        ax.grid(True, which='both', ls='--')
    plt.suptitle(f"Log-Log Plots Translated by $T_c$ (Average $T_c$ = {Tc_avg:.3f})", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_observable_errors(observables_data):
    err_keys = {"energy": "e_errs",
                "magnetisation": "mag_errs",
                "susceptibility": "sus_errs",
                "specific_heat": "s_heat_errs"}
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    ax = axs[0, 0]
    for N, data in observables_data.items():
        ax.scatter(data["t_vals"], data[err_keys["energy"]], label=f"N={N}")
    ax.set_xlabel("Temperature T")
    ax.set_ylabel("Energy Error")
    ax.set_title("Energy Error vs. Temperature")
    ax.legend()
    ax.grid(True)
    ax = axs[0, 1]
    for N, data in observables_data.items():
        ax.scatter(data["t_vals"], data[err_keys["magnetisation"]], label=f"N={N}")
    ax.set_xlabel("Temperature T")
    ax.set_ylabel("Magnetisation Error")
    ax.set_title("Magnetisation Error vs. Temperature")
    ax.legend()
    ax.grid(True)
    ax = axs[1, 0]
    for N, data in observables_data.items():
        ax.scatter(data["t_vals"], data[err_keys["susceptibility"]], label=f"N={N}")
    ax.set_xlabel("Temperature T")
    ax.set_ylabel("Susceptibility Error")
    ax.set_title("Susceptibility Error vs. Temperature")
    ax.legend()
    ax.grid(True)
    ax = axs[1, 1]
    for N, data in observables_data.items():
        ax.scatter(data["t_vals"], data[err_keys["specific_heat"]], label=f"N={N}")
    ax.set_xlabel("Temperature T")
    ax.set_ylabel("Specific Heat Error")
    ax.set_title("Specific Heat Error vs. Temperature")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    folder = "data20_3d_ising_sparse_"
    params_filepath = os.path.join(folder, "simulation_parameters.npz")
    if os.path.exists(params_filepath):
        params = np.load(params_filepath)
        nt = int(params["nt"])
        n_list = params["n_list"].tolist()
        eqSteps = int(params["eqSteps"])
        mcSteps = int(params["mcSteps"])
        T_arr = params["T_arr"]
        print("Loaded Simulation Parameters:")
        print("nt       =", nt)
        print("n_list   =", n_list)
        print("eqSteps  =", eqSteps)
        print("mcSteps  =", mcSteps)

    else:
        print(f"Simulation parameters file not found in {folder}. Using default sizes.")
        n_list = [8, 16, 32]
    compute_analysis(input_folder=folder, output_folder=folder, sizes=n_list)
    analysis_data = load_analysis(folder=folder, sizes=n_list)
    plot_binder_values(analysis_data)
    intersections = calculate_all_intersections(analysis_data)
    # save_intersections(intersections, filename="critical_intersections.npz"), 2D 2.25-2.27
    plot_all_intersections(analysis_data, intersections, zoom_margin=0.005, custom_T_low=2.25, custom_T_high=2.27)
    critical_temps, ref_size = calculate_critical_temperatures(analysis_data)
    plot_critical_temperatures(critical_temps, ref_size)
    observables_data = load_observables(input_folder=folder, sizes=n_list)
    plot_observables(observables_data)
    plot_loglog_observables(observables_data, intersections, cutoff=1e-1)
    plot_observable_errors(observables_data)
