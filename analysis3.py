# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 14:14:04 2025

@author: gianc
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import glob
import natsort
import os
from scipy.interpolate import UnivariateSpline
from scipy.optimize import brentq


# ==========================================================
# EXISTING ANALYSIS FUNCTIONS (Energy, Magnetisation, Binder, etc.)
# ==========================================================
def compute_analysis(input_folder, output_folder, sizes, dim=2):
    """
    Compute and save analysis for each system size.
    This function averages observables (energy, magnetisation, Binder cumulant, etc.)
    over simulation runs at each temperature.
    
    Now we also expect the simulation files to include the connected correlation
    function saved under the key "correlation".
    
    The magnetisation is normalized by the total number of spins (N**dim).
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    for N in sizes:
        analysis_file = os.path.join(output_folder, f"Analysis_N{N}.npz")
        if os.path.exists(analysis_file):
            print(f"Analysis file for N={N} already exists. Skipping computation.")
            continue
        files = natsort.natsorted(glob.glob(f"{input_folder}/*N{N}*.npz"))
        
        print(f"Processing system size N={N}")
        
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
        corr_list = []  
        
        for f in files:
            data = np.load(f)
            try:
                T = float(f.split("T")[1].split("N")[0])
            except Exception as e:
                print(f"Error parsing temperature from {f}: {e}")
                continue
            
            # Normalize magnetisation by total number of spins: N**dim
            norm_factor = N**dim
           
            mag = np.abs(data['magnetisation']) / norm_factor
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
            binder = 1 - m4 / (3 * m2**2)
            binder_err = np.sqrt(((2 * m4 / (3 * m2**3)) * err_m2)**2 + ((1 / (3 * m2**2)) * err_m4)**2)
            
            e = data['energy'] / norm_factor
            n_e = len(e)
            e_mean = np.mean(e)
            e_std = np.std(e, ddof=1)
            e_err = e_std / np.sqrt(n_e)
            
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
            corr_list.append(data["connected_correlation"])

        sort_idx = np.argsort(t_vals)
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
                 s_heat_errs=np.array(s_heat_errs)[sort_idx],
                 correlation=np.array(corr_list)[sort_idx])
        
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
                              "s_heat_errs": data["s_heat_errs"],
                              "corr": data["correlation"] 
                              }
        else:
            print(f"Analysis file not found for N={N}.")
    return analysis_data

def load_observables(input_folder="data6", sizes=[8, 16, 32]):

    observables_data = {}
    for N in sizes:
        filename = os.path.join(input_folder, f"Analysis_N{N}.npz")
        if os.path.exists(filename):
            data = np.load(filename)
            observables_data[N] = {
                "t_vals": data["t_vals"],
                "e_means": data["e_means"],
                "e_errs": data["e_errs"],
                "mag_means": data["mag_means"],
                "mag_errs": data["mag_errs"],
                "sus_vals": data["sus_vals"],
                "sus_errs": data["sus_errs"],
                "s_heat_vals": data["s_heat_vals"],
                "s_heat_errs": data["s_heat_errs"],
                "corr": data["correlation"] 
            }
        else:
            print(f"Analysis file not found for N={N}.")
    return observables_data

# --- Modified Binder Plot Function ---
def plot_binder_values(analysis_data, custom_T_low=None, custom_T_high=None):
    """
    Plot the Binder cumulant vs Temperature.
    Optionally, set a custom temperature range using custom_T_low and custom_T_high.
    """
    plt.figure(figsize=(8, 6))
    for N, data in analysis_data.items():
        plt.errorbar(data["t_vals"], data["binder_vals"], yerr=data["binder_errs"],
                     fmt='o', label=f"N={N}", ecolor='black', capsize=5, alpha=0.5)
    plt.axvline(x=2.269, color='red', linestyle='--', label="Expected $T_c$ (2.269)")
    if custom_T_low is not None and custom_T_high is not None:
        plt.xlim(custom_T_low, custom_T_high)
    plt.xlabel("Temperature T")
    plt.ylabel("Binder Cumulant")
    plt.legend()
    plt.grid(True)
    plt.title("Binder Cumulant vs. Temperature")
    plt.show()

# ==========================================================
# NEW FUNCTIONS FOR CORRELATION FUNCTION ANALYSIS
# ==========================================================



def plot_corr_linear(observables_data, T_targets=(2.269, 2.3)):
    """
    Plot the connected correlation function C_conn(r) vs. normalized distance (r/N)
    on a linear scale for each system size and for two target temperatures.
    
    For each system size:
      - Both target temperatures are plotted using the same base color.
      - The first target (T_targets[0]) markers are plotted with full opacity,
        and its polynomial fit is drawn as a solid line.
      - The second target (T_targets[1]) markers are plotted with reduced opacity,
        and its polynomial fit is drawn as a dotted line (with full opacity).
      - Only the fitted lines are included in the legend.
      
    Parameters:
      observables_data: Dictionary containing observables (including "corr" and "t_vals")
                        for each system size.
      T_targets: Tuple or list of two target temperatures to be plotted.
    """
    plt.figure(figsize=(8, 6))
    base_colors = plt.cm.tab10.colors  # Use Tab10 palette for distinct system sizes.
    num_base_colors = len(base_colors)
    
    for i, (N, data) in enumerate(sorted(observables_data.items())):
        if data["corr"] is None:
            continue
        base_color = base_colors[i % num_base_colors]
        
        for j, T in enumerate(T_targets):
            # Set marker transparency and line style based on the target temperature.
            if j == 0:
                marker_alpha = 1.0
                line_style = '-'  # solid line for first target.
            else:
                marker_alpha = 0.5
                line_style = ':'  # dotted line for second target.
            
            # Select the run closest to the target temperature.
            idx = np.argmin(np.abs(data["t_vals"] - T))
            corr = data["corr"][idx]  # array of C_conn(r)
            r_norm = np.arange(len(corr)) / N  # Normalize distance: r/N
            
            # Plot raw data points (markers only) without any legend label.
            plt.plot(r_norm, corr, 'o', color=base_color, alpha=marker_alpha)
            
            # Perform a polynomial (degree 10) fit to the data points.
            coeffs = np.polyfit(r_norm, corr, 10)
            p = np.poly1d(coeffs)
            x_fit = np.linspace(np.min(r_norm), np.max(r_norm), 100)
            y_fit = p(x_fit)
            # Plot the fitted curve with full opacity and add label only for the line.
            plt.plot(x_fit, y_fit, linestyle=line_style, color=base_color, alpha=1.0,
                     label=f" N={N}, T={data['t_vals'][idx]:.3f}")
    
    plt.xlabel("Normalized distance (r/N)")
    plt.ylabel("Connected Correlation Function $C_{conn}(r)$")
    plt.title("Connected Correlation Function vs. Normalized Distance (Linear Scale)")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_corr_semilog(observables_data, T_target=2.269):
    """
    Plot C_conn(r) vs. r on a semilog scale to help extract the exponential decay (for T > Tc).
    """
    plt.figure(figsize=(8,6))
    for N, data in observables_data.items():
        if data["corr"] is None:
            continue
        idx = np.argmin(np.abs(data["t_vals"] - T_target))
        corr = data["corr"][idx]
        r = np.arange(len(corr))
        plt.semilogy(r, corr, 'o-', label=f"N={N}, T={data['t_vals'][idx]:.3f}")
    plt.xlabel("Distance r")
    plt.ylabel("Connected Correlation Function $C_{conn}(r)$ (log scale)")
    plt.title("Semilog Plot of $C_{conn}(r)$ vs. r")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_corr_loglog(observables_data, T_target=2.269):
    """
    Plot C_conn(r) vs. r on a log-log scale to verify power-law decay at criticality.
    """
    plt.figure(figsize=(8,6))
    for N, data in observables_data.items():
        if data["corr"] is None:
            continue
        idx = np.argmin(np.abs(data["t_vals"] - T_target))
        corr = data["corr"][idx]
        # Avoid r=0 since log(0) is undefined; start from r=1
        r = np.arange(1, len(corr))
        plt.loglog(r, corr[1:], 'o-', label=f"N={N}, T={data['t_vals'][idx]:.3f}")
    plt.xlabel("Distance r (log scale)")
    plt.ylabel("$C_{conn}(r)$ (log scale)")
    plt.title("Log-Log Plot of $C_{conn}(r)$ vs. r")
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.show()

def compute_correlation_length(corr, N):
    """
    Compute the correlation length xi via the second-moment estimator.
    For a lattice of size N, assume that corr is the 1D array (r=0,...,R) of the connected correlation function.
    Uses:
        xi = 1/(2 sin(pi/N)) * sqrt( S(0)/S(k_min) - 1 )
    where S(0) and S(k_min) are computed with appropriate weighting.
    """
    R = len(corr) - 1  # Maximum distance index (expected to be N//2 for even N)
    
    if N % 2 == 0:
        # For even N, the maximum r is N/2 = R.
        S0 = corr[0] + 2 * np.sum(corr[1:R]) + corr[R]
        # k_min corresponds to 2pi/N. Weight the r=R term by cos(pi) = -1.
        S_k = corr[0] + 2 * np.sum(corr[1:R] * np.cos(2*np.pi*np.arange(1, R)/N)) + corr[R]*np.cos(np.pi)
    else:
        # For odd N, sum over all distances r=0,...,R.
        S0 = corr[0] + 2 * np.sum(corr[1:R+1])
        S_k = corr[0] + 2 * np.sum(corr[1:R+1] * np.cos(2*np.pi*np.arange(1, R+1)/N))
    
    # Compute the ratio in the second-moment estimator.
    ratio = S0/S_k - 1
    if ratio < 0:
        # Print a warning to help with debugging.
        print(f"Warning: Negative ratio encountered for N={N}: ratio = {ratio}, S0 = {S0}, S_k = {S_k}")
        return np.nan  # Return NaN to indicate an issue.
    
    xi = 1.0/(2*np.sin(np.pi/N)) * np.sqrt(ratio)
    return xi

def plot_correlation_length(observables_data):
    """
    For each system size, compute the correlation length xi for every temperature from the 
    connected correlation function, then plot xi vs. temperature.
    """
    plt.figure(figsize=(8,6))
    for N, data in observables_data.items():
        if data["corr"] is None:
            continue
        t_vals = data["t_vals"]
        xi_vals = []
        # Assuming data["corr"] is an array of shape (n_t, R+1)
        for corr in data["corr"]:
            xi_vals.append(compute_correlation_length(corr, N))
        xi_vals = np.array(xi_vals)
        plt.plot(t_vals, xi_vals, 'o-', label=f"N={N}")
    plt.xlabel("Temperature T")
    plt.ylabel("Correlation Length $\\xi$")
    plt.title("Correlation Length vs. Temperature")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_fss_xi(observables_data, T_target=2.269):
    """
    For runs at a temperature near T_target, plot xi vs. system size N on a log-log scale.
    The scaling relation xi ~ N^(1/nu) can be fitted to extract nu.
    """
    sizes = []
    xi_target = []
    for N, data in observables_data.items():
        if data["corr"] is None:
            continue
        idx = np.argmin(np.abs(data["t_vals"] - T_target))
        xi = compute_correlation_length(data["corr"][idx], N)
        sizes.append(N)
        xi_target.append(xi)
    sizes = np.array(sizes)
    xi_target = np.array(xi_target)
    plt.figure(figsize=(8,6))
    plt.loglog(sizes, xi_target, 'o-', label=f"T={T_target:.3f}")
    # Fit a line to log-log data to estimate 1/nu
    slope, intercept = np.polyfit(np.log(sizes), np.log(xi_target), 1)
    plt.loglog(sizes, np.exp(intercept)*sizes**slope, '--', label=f"Fit: slope = {slope:.2f}")
    plt.xlabel("System Size N")
    plt.ylabel("Correlation Length $\\xi$")
    plt.title("Finite-Size Scaling of Correlation Length")
    plt.legend()
    plt.grid(True, which='both', ls='--')
    plt.show()

def plot_scaling_collapse(observables_data, T_target=2.269, eta=0.25, d=2):
    """
    Produce a scaling collapse plot for the connected correlation function.
    At T_c, finite-size scaling theory predicts:
        C_conn(r) = N^{-(d-2+eta)} f(r/N)
    So if we plot N^(d-2+eta)*C_conn(r) vs. r/N, data for different N should collapse.
    """
    plt.figure(figsize=(8,6))
    for N, data in observables_data.items():
        if data["corr"] is None:
            continue
        idx = np.argmin(np.abs(data["t_vals"] - T_target))
        corr = data["corr"][idx]
        r = np.arange(len(corr))
        scale_factor = N**(d-2+eta)  # for 2D, this is N^(eta)
        plt.plot(r/float(N), scale_factor * corr, 'o-', label=f"N={N}")
    plt.xlabel("Scaled distance r/N")
    plt.ylabel(f"Scaled Correlation: $N^{{d-2+\\eta}} C_{{conn}}(r)$")
    plt.title(f"Scaling Collapse of Correlation Function at T={T_target:.3f}")
    plt.legend()
    plt.grid(True)
    plt.show()

# ==========================================================
# REMAINING ANALYSIS FUNCTIONS (splines, intersections, etc.)
# ==========================================================
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

# MODIFIED: Added custom_T_low and custom_T_high parameters
def calculate_all_intersections(analysis_data, s=0, custom_T_low=None, custom_T_high=None):
    splines = build_splines(analysis_data, s=s)
    intersection_dict = {}
    sizes = sorted(analysis_data.keys())
    for i in range(len(sizes)):
        for j in range(i+1, len(sizes)):
            N1 = sizes[i]
            N2 = sizes[j]
            
            # Use custom temperature range if provided
            if custom_T_low is not None and custom_T_high is not None:
                T_min = custom_T_low
                T_max = custom_T_high
            else:
                # Otherwise use the overlap of available temperature ranges
                T_min = max(np.min(analysis_data[N1]["t_vals"]), np.min(analysis_data[N2]["t_vals"]))
                T_max = min(np.max(analysis_data[N1]["t_vals"]), np.max(analysis_data[N2]["t_vals"]))
                
            Tc, Tc_error = find_intersection(splines[N1]["spline"], splines[N2]["spline"],
                                             splines[N1]["error_spline"], splines[N2]["error_spline"],
                                             T_min, T_max)
            if Tc is not None:
                intersection_dict[(N1, N2)] = (Tc, Tc_error)
                print(f"Intersection for N={N1} and N={N2}: Tc = {Tc:.4f} ± {Tc_error:.4f}")
            else:
                print(f"No intersection found for N={N1} and N={N2} in range [{T_min:.4f}, {T_max:.4f}].")
    return intersection_dict

# MODIFIED: Added custom_T_low and custom_T_high parameters
def calculate_critical_temperatures(analysis_data, custom_T_low=None, custom_T_high=None):
    splines = build_splines(analysis_data)
    reference_size = min(analysis_data.keys())
    ref_spline = splines[reference_size]["spline"]
    ref_error_spline = splines[reference_size]["error_spline"]
    critical_temps = {}
    for N, spline_data in splines.items():
        if N == reference_size:
            continue
            
        # Use custom temperature range if provided
        if custom_T_low is not None and custom_T_high is not None:
            T_min = custom_T_low
            T_max = custom_T_high
        else:
            # Otherwise use the overlap of available temperature ranges
            T_min = max(np.min(analysis_data[reference_size]["t_vals"]), np.min(analysis_data[N]["t_vals"]))
            T_max = min(np.max(analysis_data[reference_size]["t_vals"]), np.max(analysis_data[N]["t_vals"]))
            
        Tc, Tc_error = find_intersection(ref_spline, spline_data["spline"],
                                         ref_error_spline, spline_data["error_spline"],
                                         T_min, T_max)
        if Tc is not None:
            critical_temps[N] = (Tc, Tc_error)
            print(f"Intersection for N={N} (ratio {N/reference_size:.2f}): Tc = {Tc:.4f} ± {Tc_error:.4f}")
        else:
            print(f"No intersection found for N={N} in range [{T_min:.4f}, {T_max:.4f}].")
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

def load_observables_extended(input_folder="data6", sizes=[8, 16, 32]):
    """
    This function loads all observables including the connected correlation function.
    """
    observables_data = {}
    for N in sizes:
        filename = os.path.join(input_folder, f"Analysis_N{N}.npz")
        if os.path.exists(filename):
            data = np.load(filename)
            observables_data[N] = {
                "t_vals": data["t_vals"],
                "e_means": data["e_means"],
                "e_errs": data["e_errs"],
                "mag_means": data["mag_means"],
                "mag_errs": data["mag_errs"],
                "sus_vals": data["sus_vals"],
                "sus_errs": data["sus_errs"],
                "s_heat_vals": data["s_heat_vals"],
                "s_heat_errs": data["s_heat_errs"],
                "corr": data["correlation"] if "correlation" in data else None
            }
        else:
            print(f"Analysis file not found for N={N}.")
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
    
def plot_loglog_observables(observables_data, intersections, cutoff=1e-5, Tc_avg=2.269, 
                             reduced_range_low=None, reduced_range_high=None, T_target=2.269,
                             side='lower'):
    """
    Produce a 2x2 grid of log–log plots for critical exponent analysis.

    The first three subplots show observables vs. the reduced temperature.
    Depending on the value of the `side` argument:
      - For side='lower':  t = (Tc - T) / Tc   (using data for T < Tc)
      - For side='upper':  t = (T - Tc) / Tc   (using data for T > Tc)
    
    The fourth subplot shows the correlation length (computed from the correlation function)
    plotted versus system size on a log–log scale, at a chosen target temperature T_target.
    
    Parameters:
      observables_data: Dictionary containing t_vals, observable arrays, and correlation data for each system size.
      intersections: (Unused in this function; kept for consistency.)
      cutoff: Minimum value for the reduced temperature (to avoid numerical issues near 0).
      Tc_avg: Critical temperature to use for computing the reduced variable.
      reduced_range_low, reduced_range_high: Custom bounds for the reduced temperature.
      T_target: The target temperature at which to compute the correlation length for each system size.
      side: String, either 'lower' (default) to use T < Tc data or 'upper' to use T > Tc data.
    """


    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    
    # The observables to plot on the first three subplots:
    obs_keys = {"magnetisation": "mag_means",
                "susceptibility": "sus_vals",
                "specific_heat": "s_heat_vals"}
    obs_titles = {"magnetisation": "Magnetisation", 
                  "susceptibility": "Susceptibility", 
                  "specific_heat": "Specific Heat"}
    
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    ax_idx = 0
    
    # Set the appropriate x-label and suptitle fragment depending on side.
    if side.lower() == 'lower':
        xlabel = r"$\frac{T_c-T}{T_c}$"
    elif side.lower() == 'upper':
        xlabel = r"$\frac{T-T_c}{T_c}$"

    
    # Plot the three observables vs. reduced temperature.
    for obs, key in obs_keys.items():
        ax = axs.flat[ax_idx]
        for j, N in enumerate(sorted(observables_data.keys())):
            data = observables_data[N]
            T = data["t_vals"]
            y = data[key]
            # For magnetisation, take the absolute value.
            if obs == "magnetisation":
                y = np.abs(y)
            
            # Compute the reduced temperature based on the chosen side.
            if side.lower() == 'lower':
                # Use data for T < Tc: t = (Tc - T) / Tc.
                x = (Tc_avg - T) / Tc_avg
            elif side.lower() == 'upper':
                # Use data for T > Tc: t = (T - Tc) / Tc.
                x = (T - Tc_avg) / Tc_avg
            
            # Apply the cutoff and the custom reduced range limits.
            mask = x > cutoff
            if reduced_range_low is not None:
                mask = mask & (x >= reduced_range_low)
            if reduced_range_high is not None:
                mask = mask & (x <= reduced_range_high)
            x_plot, y_plot = x[mask], y[mask]
            if len(x_plot) < 2:
                continue
            color = colors[j % len(colors)]
            ax.errorbar(x_plot, y_plot, fmt='o', color=color, label=f"N={N}",
                        ecolor='black', capsize=5, alpha=0.5)
            # Fit a linear relation on the log–log data.
            logx, logy = np.log(x_plot), np.log(y_plot)
            p, cov = np.polyfit(logx, logy, 1, cov=True)
            slope, intercept = p
            slope_error = np.sqrt(cov[0, 0])
            x_fit = np.linspace(np.min(x_plot), np.max(x_plot), 100)
            y_fit = np.exp(intercept) * x_fit**slope
            ax.plot(x_fit, y_fit, '--', color=color,
                    label=f"N={N} fit: {slope:.3f}±{slope_error:.3f}")
            ax.text(np.median(x_plot), np.median(y_fit), f"{slope:.3f}±{slope_error:.3f}",
                    color=color, fontsize=9, bbox=dict(facecolor='white', alpha=0.5))
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(obs_titles[obs])
        ax.set_title(f"Log-Log Plot of {obs_titles[obs]}")
        ax.legend(fontsize='small')
        ax.grid(True, which='both', ls='--')
        ax_idx += 1
        
    # Fourth subplot: Correlation Length vs. System Size (Finite–Size Scaling).
    ax = axs.flat[ax_idx]
    sizes = []
    xi_target = []
    for N in sorted(observables_data.keys()):
        data = observables_data[N]
        if data["corr"] is None:
            continue
        # Pick the index corresponding to the temperature closest to T_target.
        idx = np.argmin(np.abs(data["t_vals"] - T_target))
        # Compute the correlation length using your compute_correlation_length function.
        xi_val = compute_correlation_length(data["corr"][idx], N)
        sizes.append(N)
        xi_target.append(xi_val)
    sizes = np.array(sizes)
    xi_target = np.array(xi_target)
    ax.loglog(sizes, xi_target, 'o-', label=f"T={T_target:.3f}")
    # Fit a line to log–log data to extract the finite–size scaling exponent (1/ν).
    if len(sizes) > 1:
        slope, intercept = np.polyfit(np.log(sizes), np.log(xi_target), 1)
        ax.loglog(sizes, np.exp(intercept)*sizes**slope, '--', label=f"Fit: slope = {slope:.2f}")
    ax.set_xlabel("System Size N")
    ax.set_ylabel("Correlation Length $\\xi$")
    ax.set_title("Finite-Size Scaling: $\\xi$ vs N")
    ax.legend()
    ax.grid(True, which='both', ls='--')
    
    plt.suptitle("Log-Log Critical Exponent Analysis", fontsize=16)
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
    
def plot_Cv_fss(observables_data, Tc=2.269):
    """
    Plot the maximum specific heat C_v^{max} versus system size L on a semi-log scale.
    For the 2D Ising model, we expect: C_v^{max} ~ A + B * ln(L)
    """
    sizes = []
    Cv_max = []
    
    for N, data in observables_data.items():
        # Only consider data near the critical region
        # For example, you might restrict T values to a small window around Tc
        mask = (data["t_vals"] > (Tc - 0.1)) & (data["t_vals"] < (Tc + 0.1))
        if np.sum(mask) < 1:
            continue
        Cv = data["s_heat_vals"][mask]
        sizes.append(N)
        Cv_max.append(np.max(Cv))
    
    sizes = np.array(sizes)
    Cv_max = np.array(Cv_max)
    
    plt.figure(figsize=(8,6))
    plt.plot(np.log(sizes), Cv_max, 'o-', label="Data")
    # Fit a linear relation: C_v^{max} = A + B*ln(L)
    A, B = np.polyfit(np.log(sizes), Cv_max, 1)
    plt.plot(np.log(sizes), A + B*np.log(sizes), '--', label=f"Fit: C_v^max = {A:.2f} + {B:.2f} ln(L)")
    plt.xlabel("ln(System Size L)")
    plt.ylabel("Maximum Specific Heat $C_v^{max}$ ($k_B$ per site)")
    plt.title("Finite-Size Scaling of Specific Heat\n(Logarithmic divergence: α = 0)")
    plt.legend()
    plt.grid(True)
    plt.show()
    
def plot_semi_log_specific_heat(observables_data, Tc=2.269, T_range=None):
    """
    Produce a semi-log plot for the specific heat as a function of ln|T-Tc|
    to illustrate the logarithmic divergence expected in the 2D Ising model (α = 0).
    
    Parameters:
      observables_data: Dictionary with keys for each system size (e.g., N=16, 32, 64, 128)
                        Each value is a dict containing arrays for "t_vals" and "s_heat_vals".
      Tc: Critical temperature (default: 2.269 in J/k_B)
      T_range: Optional tuple (T_min, T_max) to restrict the T values used for plotting.
               If None, all available T values are used.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    
    for N, data in sorted(observables_data.items()):
        T_vals = data["t_vals"]
        Cv_vals = data["s_heat_vals"]
        
        # Optionally restrict to a specific T range (e.g., near Tc)
        if T_range is not None:
            mask = (T_vals >= T_range[0]) & (T_vals <= T_range[1])
            T_vals = T_vals[mask]
            Cv_vals = Cv_vals[mask]
        
        # Compute the absolute difference |T - Tc|
        diff = np.abs(T_vals - Tc)
        # Avoid log(0) by replacing zeros with a small number
        diff[diff == 0] = 1e-10
        ln_diff = np.log(diff)
        
        # Plot Cv versus ln(|T-Tc|)
        plt.plot(ln_diff, Cv_vals, 'o-', label=f"N={N}")
    
    plt.xlabel("ln|T - Tc| (dimensionless)")
    plt.ylabel("Specific Heat $C_v$ ($k_B$ per site)")
    plt.title("Semi-log Plot of Specific Heat vs. ln|T-Tc|\n(Logarithmic divergence: α = 0)")
    plt.legend()
    plt.grid(True)
    plt.show()    
    
def plot_mag_vs_iterations_comparison(primary_folder, secondary_folder, temperature, system_size, max_iterations=None):
    """
    For a given temperature and system size, search for a file in each of two folders
    (e.g., one containing a Wolfs algorithm run and the other a Metropolis algorithm run).
    Load the magnetisation time series from each file and plot magnetisation vs iterations
    on the same plot with different labels.
    
    Parameters:
      primary_folder   : Folder containing simulation files from the first algorithm (e.g., Wolfs).
      secondary_folder : Folder containing simulation files from the second algorithm (e.g., Metropolis).
      temperature      : The temperature at which to search for files (the filename should contain 'T<temperature>').
      system_size      : The system size to filter files (the filename should contain 'N<system_size>').
      max_iterations   : Maximum number of iterations to plot (if None, plot all iterations).
    """
    # Create a file search pattern for the given temperature and system size
    primary_pattern = os.path.join(primary_folder, f"*T{temperature}*N{system_size}*.npz")
    primary_files = natsort.natsorted(glob.glob(primary_pattern))
    if not primary_files:
        print(f"No file found in {primary_folder} for T={temperature} and N={system_size}")
        return
    primary_file = primary_files[0]
    data_primary = np.load(primary_file)
    mag_primary = data_primary['magnetisation']
    
    secondary_pattern = os.path.join(secondary_folder, f"*T{temperature}*N{system_size}*.npz")
    secondary_files = natsort.natsorted(glob.glob(secondary_pattern))
    if not secondary_files:
        print(f"No file found in {secondary_folder} for T={temperature} and N={system_size}")
        return
    secondary_file = secondary_files[0]
    data_secondary = np.load(secondary_file)
    mag_secondary = data_secondary['magnetisation']
    
    # Limit iterations if max_iterations is provided
    if max_iterations is not None:
        mag_primary = mag_primary[:max_iterations]
        mag_secondary = mag_secondary[:max_iterations]
    
    iterations_primary = np.arange(len(mag_primary))
    iterations_secondary = np.arange(len(mag_secondary))
    
    plt.figure(figsize=(8, 6))
    plt.plot(iterations_primary, mag_primary, label="Wolfs Algorithm", marker='o', linestyle='-')
    plt.plot(iterations_secondary, mag_secondary, label="Metropolis Algorithm", marker='s', linestyle='--')
    plt.xlabel("Iteration")
    plt.ylabel("Magnetisation")
    plt.title(f"Magnetisation vs Iterations at T={temperature}, N={system_size}")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    folder = "final_cluster_exponent"  # Change folder if needed
    params_filepath = os.path.join(folder, "simulation_parameters.npz")
    if os.path.exists(params_filepath):
        params = np.load(params_filepath)
        nt = int(params["nt"])
        n_list = params["n_list"].tolist() #[16, 32, 64, 128]
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
    compute_analysis(input_folder=folder, output_folder=folder, sizes=n_list, dim=2)
    analysis_data = load_analysis(folder=folder, sizes=n_list)
    
    # Adjust the Binder plot range by setting custom_T_low and custom_T_high here:
    # binder_range_low = 2.26
    # binder_range_high = 2.3
    
    binder_range_low = 2.1
    binder_range_high = 2.5
    
    # Set temperature range for intersection calculations
    intersection_range_low = 2.26
    intersection_range_high = 2.27
   
   # Analyze Binder cumulant noise at high temperatures
    high_T_threshold = 3.0
    
    # binder_range_low = 4.4
    # binder_range_high = 4.6
    plot_binder_values(analysis_data, custom_T_low=binder_range_low, custom_T_high=binder_range_high)
    
    
    intersections = calculate_all_intersections(analysis_data, 
                                               custom_T_low=intersection_range_low, 
                                               custom_T_high=intersection_range_high)
    # plot_all_intersections(analysis_data, intersections, zoom_margin=0.005, custom_T_low=2.2, custom_T_high=2.3)
    
    plot_all_intersections(analysis_data, intersections, zoom_margin=0.005, custom_T_low=2.2685, custom_T_high=2.2695)
    # plot_all_intersections(analysis_data, intersections, zoom_margin=0.005, custom_T_low=4.5, custom_T_high=4.515)
    critical_temps, ref_size = calculate_critical_temperatures(analysis_data, custom_T_low=intersection_range_low, 
    custom_T_high=intersection_range_high )
    
    plot_critical_temperatures(critical_temps, ref_size)
    
    observables_data = load_observables_extended(input_folder=folder, sizes=n_list)
    
    plot_observables(observables_data)
    
    T_target = 2.26
    plot_corr_linear(observables_data, T_targets=(2.269, 3))


    plot_correlation_length(observables_data)
    #plot_fss_xi(observables_data, T_target=T_target)
    # For scaling collapse, specify eta (for example, 0.25 for 2D Ising) and dimensionality d

    plot_loglog_observables(observables_data, intersections, Tc_avg=2.269, 
                        reduced_range_low=0.005, reduced_range_high=1, side='lower')


    # plot_observable_errors(observables_data)
    #plot_Cv_fss(observables_data, Tc=2.269)
    #plot_semi_log_specific_heat(observables_data, Tc=2.269, T_range=(2, 2.269))

    plot_mag_vs_iterations_comparison(
        primary_folder="wolfs1",
        secondary_folder="metropolis1",
        temperature=2.275,
        system_size=32,
        max_iterations=500  # Only the first 500 iterations will be plotted
    )