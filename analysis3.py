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
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def plot_binder_values_with_critical(analysis_data, critical_temps, reference_size, 
                                     custom_T_low=None, custom_T_high=None,
                                     inset_xlim=None, inset_ylim=None,
                                     binder_text="(a)", binder_text_pos=(0.05, 0.95),
                                     inset_text="(y)", inset_text_pos=(0.05, 0.90)):
    """
    Plots the Binder cumulant vs. Temperature for each system size and then
    overlays an inset axes (positioned in the bottom left) showing the estimated 
    critical temperatures (with error bars) vs. the system size ratio. In the inset,
    only the three critical points are shown.
    
    Parameters:
      analysis_data : dict
          Dictionary containing analysis data (including t_vals, binder_vals, etc.)
          for each system size.
      critical_temps : dict
          Dictionary mapping system sizes to tuples (Tc, Tc_error).
      reference_size : int
          The smallest system size used as reference to compute ratios.
      custom_T_low, custom_T_high : float, optional
          Limits for the main plot x–axis (Temperature T).
      inset_xlim : tuple (xmin, xmax), optional
          Manually specify the x–axis limits for the inset (system size ratio).
      inset_ylim : tuple (ymin, ymax), optional
          Manually specify the y–axis limits for the inset (critical temperature).
      binder_text : str, optional
          Text label to add on the main Binder plot.
      binder_text_pos : tuple (x, y), optional
          Position for the main Binder plot text in axes coordinates (0-1 range).
      inset_text : str, optional
          Text label to add on the inset (critical temperature) plot.
      inset_text_pos : tuple (x, y), optional
          Position for the inset text in axes coordinates (0-1 range).
    """
    plt.figure(figsize=(8, 6))
    for N, data in analysis_data.items():
        plt.errorbar(data["t_vals"], data["binder_vals"], yerr=data["binder_errs"],
                     fmt='o', label=f"L={N}", ecolor='black', capsize=5, alpha=0.5)
    plt.axvline(x=2.269, color='red', linestyle='--', label="Expected $\mathbf{T_c}$ (2.269)")
    if custom_T_low is not None and custom_T_high is not None:
        plt.xlim(custom_T_low, custom_T_high)
    plt.xlabel("Temperature, $\mathbf{T}$", fontsize=16)
    plt.ylabel("Binder Cumulant, $\mathbf{U_L}$", fontsize=16)
    plt.legend(fontsize=12, loc="upper right")
    plt.tick_params(axis='both', which='major', labelsize=14)

    plt.gca().text(binder_text_pos[0], binder_text_pos[1], binder_text,
                   transform=plt.gca().transAxes, fontsize=20, fontweight='bold')
    
    ax_inset = inset_axes(plt.gca(), width="100%", height="100%", 
                          bbox_to_anchor=(0.11, 0.09, 0.4, 0.5),
                          bbox_transform=plt.gca().transAxes, borderpad=0)
    
    ratios = [N / reference_size for N in critical_temps.keys()]
    Tc_values = [val[0] for val in critical_temps.values()]
    Tc_errors = [val[1] for val in critical_temps.values()]
    
    ax_inset.errorbar(ratios, Tc_values, yerr=Tc_errors, fmt='o', color='blue',
                      ecolor='black', capsize=4, alpha=0.8, label="Estimated $\mathbf{T_c}$")
    
    avg_Tc = np.mean(Tc_values)
    avg_error = np.mean(Tc_errors)
    ax_inset.axhline(y=avg_Tc, color='blue', linestyle='--',
                     label="Avg $\mathbf{T_c}$ = " + f"{avg_Tc:.3f}±{avg_error:.3f}")
    
    ax_inset.axhline(y=2.269, color='red', linestyle='--', label="Expected $\mathbf{T_c}$")
    
    if inset_xlim is not None:
        ax_inset.set_xlim(inset_xlim)
    else:
        margin = 0.1
        ax_inset.set_xlim(min(ratios) - margin, max(ratios) + margin)
    
    if inset_ylim is not None:
        ax_inset.set_ylim(inset_ylim)
    
    ax_inset.set_xticks(ratios)
    ax_inset.set_xlabel("$\mathbf{L / L_{min}}$", fontsize=16)
    ax_inset.xaxis.set_label_coords(0.5, -0.03)
    ax_inset.set_ylabel("$\mathbf{T}$", fontsize=16, rotation=0)
    ax_inset.yaxis.set_label_coords(-0.07, 1.03)
    ax_inset.xaxis.label.set_bbox(dict(facecolor='white', edgecolor='black', pad=2))
    ax_inset.yaxis.label.set_bbox(dict(facecolor='white', edgecolor='black', pad=3))
    
    ax_inset.tick_params(axis='both', which='major', labelsize=11)
    ax_inset.legend(fontsize=11, loc='upper right', bbox_to_anchor=(1.00, 0.4))

    
    ax_inset.text(inset_text_pos[0], inset_text_pos[1], inset_text,
                  transform=ax_inset.transAxes, fontsize=20, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig("C:/Users/gianc/Documents/isingproject/final figs/bind_final.pdf", format="pdf", bbox_inches="tight")
    plt.show()

    
def compute_analysis(input_folder, output_folder, sizes, dim=2):
    """
    Compute and save analysis for each system size.
    This function averages observables (energy, magnetisation, Binder cumulant, etc.)
    over simulation runs at each temperature, storing them in per–spin units.

    The final arrays (e_means, mag_means, sus_vals, s_heat_vals) will be:
      e_means, mag_means:        E/N^2,  M/N^2
      sus_vals, s_heat_vals:     per–spin definitions of susceptibility & specific heat
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for N in sizes:
        analysis_file = os.path.join(output_folder, f"Analysis_N{N}.npz")
        # Remove or comment out this block if you want to overwrite existing files:
        if os.path.exists(analysis_file):
            print(f"Analysis file for N={N} already exists. Skipping computation.")
            continue

        files = natsort.natsorted(glob.glob(f"{input_folder}/*N{N}*.npz"))
        print(f"Processing system size N={N}")

        norm_factor = N**dim  

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

            M = np.abs(data["magnetisation"])
            n_mag = len(M)
            M_mean = np.mean(M)
            M_std  = np.std(M, ddof=1)
            M_err  = M_std / np.sqrt(n_mag)

            M2_mean = np.mean(M**2)
            M4_mean = np.mean(M**4)
            M2_std  = np.std(M**2, ddof=1)
            M4_std  = np.std(M**4, ddof=1)
            M2_err  = M2_std / np.sqrt(n_mag)
            M4_err  = M4_std / np.sqrt(n_mag)

            if M2_mean == 0:
                continue
            binder = 1 - M4_mean / (3 * M2_mean**2)
            binder_err = np.sqrt(
                ((2 * M4_mean / (3 * M2_mean**3)) * M2_err)**2
                + ((1 / (3 * M2_mean**2)) * M4_err)**2
            )

            E = data["energy"]
            n_e = len(E)
            E_mean = np.mean(E)
            E_std  = np.std(E, ddof=1)
            E_err  = E_std / np.sqrt(n_e)

            if T == 0:
                continue
            beta = 1.0 / T

            var_M = (M2_mean - M_mean**2)
            sus_val = beta * var_M / (norm_factor)
            var_M_err = np.sqrt(M2_err**2 + (2 * M_mean * M_err)**2)
            sus_err = beta * var_M_err / norm_factor

            E2_mean = np.mean(E**2)
            E2_std  = np.std(E**2, ddof=1)
            E2_err  = E2_std / np.sqrt(n_e)
            var_E   = (E2_mean - E_mean**2)
            s_heat_val = beta**2 * var_E / norm_factor
            var_E_err = np.sqrt(E2_err**2 + (2 * E_mean * E_err)**2)
            s_heat_err = beta**2 * var_E_err / norm_factor

            e_means.append(E_mean / norm_factor)
            e_errs.append(E_err  / norm_factor)

            mag_means.append(M_mean / norm_factor)
            mag_errs.append(M_err  / norm_factor)

            sus_vals.append(sus_val)
            sus_errs.append(sus_err)
            s_heat_vals.append(s_heat_val)
            s_heat_errs.append(s_heat_err)

            corr_list.append(data["correlation"])

            t_vals.append(T)
            binder_vals.append(binder)
            binder_errs.append(binder_err)

        sort_idx = np.argsort(t_vals)
        np.savez(
            analysis_file,
            t_vals        = np.array(t_vals)[sort_idx],
            binder_vals   = np.array(binder_vals)[sort_idx],
            binder_errs   = np.array(binder_errs)[sort_idx],
            e_means       = np.array(e_means)[sort_idx],
            e_errs        = np.array(e_errs)[sort_idx],
            mag_means     = np.array(mag_means)[sort_idx],
            mag_errs      = np.array(mag_errs)[sort_idx],
            sus_vals      = np.array(sus_vals)[sort_idx],
            sus_errs      = np.array(sus_errs)[sort_idx],
            s_heat_vals   = np.array(s_heat_vals)[sort_idx],
            s_heat_errs   = np.array(s_heat_errs)[sort_idx],
            correlation   = np.array(corr_list)[sort_idx]
        )

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

def plot_binder_values(analysis_data, custom_T_low=None, custom_T_high=None):
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
    plt.show()

def plot_corr_linear(observables_data, T_targets=(2.269, 2.3)):
    plt.figure(figsize=(8, 6))
    base_colors = plt.cm.tab10.colors  
    num_base_colors = len(base_colors)
    
    for i, (N, data) in enumerate(sorted(observables_data.items())):
        if data["corr"] is None:
            continue
        base_color = base_colors[i % num_base_colors]
        
        for j, T in enumerate(T_targets):
            if j == 0:
                marker_alpha = 1.0
                line_style = '-'
            else:
                marker_alpha = 0.5
                line_style = ':'
            
            idx = np.argmin(np.abs(data["t_vals"] - T))
            corr = data["corr"][idx]
            r_norm = np.arange(len(corr)) / N
            
            plt.plot(r_norm, corr, 'o', color=base_color, alpha=marker_alpha)
            
            coeffs = np.polyfit(r_norm, corr, 10)
            p = np.poly1d(coeffs)
            x_fit = np.linspace(np.min(r_norm), np.max(r_norm), 100)
            y_fit = p(x_fit)
            plt.plot(x_fit, y_fit, linestyle=line_style, color=base_color, alpha=1.0,
                     label=f" N={N}, T={data['t_vals'][idx]:.3f}")
    
    plt.xlabel("Normalized distance (r/N)")
    plt.ylabel("$G(r)$")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_corr_semilog(observables_data, T_target=2.269):
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
    plt.figure(figsize=(8,6))
    for N, data in observables_data.items():
        if data["corr"] is None:
            continue
        idx = np.argmin(np.abs(data["t_vals"] - T_target))
        corr = data["corr"][idx]
        r = np.arange(1, len(corr))
        plt.loglog(r, corr[1:], 'o-', label=f"N={N}, T={data['t_vals'][idx]:.3f}")
    plt.xlabel("Distance r (log scale)")
    plt.ylabel("$C_{conn}(r)$ (log scale)")
    plt.title("Log-Log Plot of $C_{conn}(r)$ vs. r")
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.show()

def compute_correlation_length(corr, N, corr_err=None):
    R = len(corr) - 1
    
    if N % 2 == 0:
        S0 = corr[0] + 2 * np.sum(corr[1:R]) + corr[R]
        S_k = (corr[0] + 2 * np.sum(corr[1:R] * np.cos(2*np.pi*np.arange(1, R)/N))
               + corr[R]*np.cos(np.pi))
        if corr_err is not None:
            S0_err = np.sqrt(corr_err[0]**2 + 4*np.sum(corr_err[1:R]**2) + corr_err[R]**2)
            S_k_err = np.sqrt(corr_err[0]**2 + 4*np.sum((np.cos(2*np.pi*np.arange(1,R)/N)**2 * corr_err[1:R]**2))
                               + (np.cos(np.pi)*corr_err[R])**2)
        else:
            S0_err = S_k_err = np.nan
    else:
        S0 = corr[0] + 2 * np.sum(corr[1:R+1])
        S_k = corr[0] + 2 * np.sum(corr[1:R+1] * np.cos(2*np.pi*np.arange(1, R+1)/N))
        if corr_err is not None:
            S0_err = np.sqrt(corr_err[0]**2 + 4*np.sum(corr_err[1:R+1]**2))
            S_k_err = np.sqrt(corr_err[0]**2 + 4*np.sum((np.cos(2*np.pi*np.arange(1,R+1)/N)**2 * corr_err[1:R+1]**2))
                               )
        else:
            S0_err = S_k_err = np.nan

    ratio = S0/S_k - 1
    if ratio < 0:
        print(f"Warning: Negative ratio encountered for N={N}: ratio = {ratio}, S0 = {S0}, S_k = {S_k}")
        return np.nan, np.nan
    xi = 1.0/(2*np.sin(np.pi/N)) * np.sqrt(ratio)
    
    if corr_err is not None:
        dR_dS0 = 1.0 / S_k
        dR_dS_k = -S0 / (S_k**2)
        ratio_err = np.sqrt((dR_dS0 * S0_err)**2 + (dR_dS_k * S_k_err)**2)
        factor = 1.0/(2*np.sin(np.pi/N))
        xi_err = factor/(2*np.sqrt(ratio)) * ratio_err
    else:
        xi_err = np.nan

    return xi, xi_err

def plot_correlation_length(observables_data):
    plt.figure(figsize=(8, 6))
    for N, data in observables_data.items():
        if data["corr"] is None:
            continue

        t_vals = data["t_vals"]
        xi_norm_vals = []
        xi_norm_errs = []

        has_corr_err = "corr_err" in data

        for i, corr_raw in enumerate(data["corr"]):
            m = data["mag_means"][i]
            corr_connected = corr_raw - m**2
            corr_err = data["corr_err"][i] if has_corr_err else None
            xi, xi_err = compute_correlation_length(corr_connected, N, corr_err)
            xi_norm_vals.append(xi / N)
            xi_norm_errs.append(xi_err / N if xi_err is not None else np.nan)

        xi_norm_vals = np.array(xi_norm_vals)
        xi_norm_errs = np.array(xi_norm_errs)
        plt.errorbar(t_vals, xi_norm_vals, yerr=xi_norm_errs, fmt='o-', capsize=5,
                     label=f"L={N}")

    plt.xlabel("Temperature,  T")
    plt.ylabel("Normalized Correlation Length, $\\xi/L$")
    plt.legend()
    plt.savefig("C:/Users/gianc/Documents/isingproject/final figs/c_length_final.pdf", format="pdf", bbox_inches="tight")
    plt.show()

def plot_fss_xi(observables_data, T_target=2.269):
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
    slope, intercept = np.polyfit(np.log(sizes), np.log(xi_target), 1)
    plt.loglog(sizes, np.exp(intercept)*sizes**slope, '--', label=f"Fit: slope = {slope:.2f}")
    plt.xlabel("System Size N")
    plt.ylabel("Correlation Length $\\xi$")
    plt.title("Finite-Size Scaling of Correlation Length")
    plt.legend()
    plt.grid(True, which='both', ls='--')
    plt.show()

def plot_scaling_collapse(observables_data, T_target=2.269, eta=0.25, d=2):
    plt.figure(figsize=(8,6))
    for N, data in observables_data.items():
        if data["corr"] is None:
            continue
        idx = np.argmin(np.abs(data["t_vals"] - T_target))
        corr = data["corr"][idx]
        r = np.arange(len(corr))
        scale_factor = N**(d-2+eta)
        plt.plot(r/float(N), scale_factor * corr, 'o-', label=f"N={N}")
    plt.xlabel("Scaled distance r/N")
    plt.ylabel(f"Scaled Correlation: $N^{{d-2+\\eta}} C_{{conn}}(r)$")
    plt.title(f"Scaling Collapse of Correlation Function at T={T_target:.3f}")
    plt.legend()
    plt.grid(True)
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

def calculate_all_intersections(analysis_data, s=0, custom_T_low=None, custom_T_high=None):
    splines = build_splines(analysis_data, s=s)
    intersection_dict = {}
    sizes = sorted(analysis_data.keys())
    for i in range(len(sizes)):
        for j in range(i+1, len(sizes)):
            N1 = sizes[i]
            N2 = sizes[j]
            
            if custom_T_low is not None and custom_T_high is not None:
                T_min = custom_T_low
                T_max = custom_T_high
            else:
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

def calculate_critical_temperatures(analysis_data, custom_T_low=None, custom_T_high=None):
    splines = build_splines(analysis_data)
    reference_size = min(analysis_data.keys())
    ref_spline = splines[reference_size]["spline"]
    ref_error_spline = splines[reference_size]["error_spline"]
    critical_temps = {}
    for N, spline_data in splines.items():
        if N == reference_size:
            continue
            
        if custom_T_low is not None and custom_T_high is not None:
            T_min = custom_T_low
            T_max = custom_T_high
        else:
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
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_critical_temperatures(critical_temps, reference_size):
    ratios = [N / reference_size for N in critical_temps.keys()]
    Tc_values = [val[0] for val in critical_temps.values()]
    Tc_errors = [val[1] for val in critical_temps.values()]
    
    plt.figure(figsize=(8, 6))
    plt.errorbar(ratios, Tc_values, yerr=Tc_errors, fmt='o', color='blue',
                 label="Estimated $T_c$", ecolor='black', capsize=5, alpha=0.7)
    
    avg_Tc = np.mean(Tc_values)
    avg_error = np.mean(Tc_errors)
    
    plt.axhline(y=avg_Tc, color='blue', linestyle='--',
                label=f"Average $T_c$ = {avg_Tc:.3f} ± {avg_error:.3f}")
    
    plt.axhline(y=2.269, color='red', linestyle='--', label="Expected $T_c$ (2.269)")
    
    plt.xlabel("System Size Ratio ($N / N_{small}$)")
    plt.ylabel("Critical Temperature $T_c$")
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
    fig, axs = plt.subplots(4, 1, figsize=(12, 16))
    
    ax = axs[0]
    for N, data in observables_data.items():
        ax.errorbar(data["t_vals"], data["e_means"], yerr=data["e_errs"],
                    fmt='o', label=f"L={N}", ecolor='black', capsize=5, alpha=0.5)
    ax.set_xlabel("Temperature, $T$")
    ax.set_ylabel("Energy, $<H>$")
    ax.legend()
    
    ax = axs[1]
    for N, data in observables_data.items():
        ax.errorbar(data["t_vals"], data["mag_means"], yerr=data["mag_errs"],
                    fmt='o', label=f"L={N}", ecolor='black', capsize=5, alpha=0.5)
    all_T = np.concatenate([data["t_vals"] for data in observables_data.values()])
    T_min = np.min(all_T)
    Tc = 2.269
    T_vals = np.linspace(T_min, Tc, 300)
    ax.plot([Tc, Tc], [0, 0.38], 'k--', lw=2)
    m_ons = (1 - np.sinh(2/T_vals)**(-4))**(1/8)
    ax.plot(T_vals, m_ons, 'k--', lw=2, label="Onsager")
    ax.set_xlabel("Temperature, $T$")
    ax.set_ylabel("Magnetisation, $M$")
    ax.legend()
    
    ax = axs[2]
    for N, data in observables_data.items():
        ax.errorbar(data["t_vals"], data["sus_vals"], yerr=data["sus_errs"],
                    fmt='o', label=f"L={N}", ecolor='black', capsize=5, alpha=0.5)
    ax.set_xlabel("Temperature, $T$")
    ax.set_ylabel("Susceptibility, $\chi$")
    ax.legend()
    
    ax = axs[3]
    for N, data in observables_data.items():
        ax.errorbar(data["t_vals"], data["s_heat_vals"], yerr=data["s_heat_errs"],
                    fmt='o', label=f"L={N}", ecolor='black', capsize=5, alpha=0.5)
    ax.set_xlabel("Temperature, $T$")
    ax.set_ylabel("Specific Heat, $C$")
    ax.legend()
    
    plt.tight_layout()
    plt.savefig("C:/Users/gianc/Documents/isingproject/final figs/obs.pdf", format="pdf", bbox_inches="tight")
    plt.show()

def plot_loglog_observables(observables_data, intersections, cutoff=1e-5, Tc_avg=2.269, 
                            reduced_range_low=None, reduced_range_high=None, T_target=2.269,
                            side='lower'):
    import numpy as np
    fig, axs = plt.subplots(4, 1, figsize=(8, 12))
    
    obs_keys = {"magnetisation": "mag_means",
                "susceptibility": "sus_vals",
                "specific_heat": "s_heat_vals"}
    err_keys = {"magnetisation": "mag_errs",
                "susceptibility": "sus_errs",
                "specific_heat": "s_heat_errs"}
    obs_titles = {"magnetisation": "Magnetisation, $M$", 
                  "susceptibility": "Susceptibility, $\\chi$", 
                  "specific_heat": "Specific Heat, $C$"}
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for i, (obs, key) in enumerate(obs_keys.items()):
        ax = axs[i]
        for j, N in enumerate(sorted(observables_data.keys())):
            data = observables_data[N]
            T = data["t_vals"]
            y = data[key]
            y_err = data[err_keys[obs]]
            if obs == "magnetisation":
                y = np.abs(y)
            if side.lower() == 'lower':
                x = (Tc_avg - T) / Tc_avg
            elif side.lower() == 'upper':
                x = (T - Tc_avg) / Tc_avg
            mask = x > cutoff
            if reduced_range_low is not None:
                mask &= (x >= reduced_range_low)
            if reduced_range_high is not None:
                mask &= (x <= reduced_range_high)
            x_plot, y_plot = x[mask], y[mask]
            y_err_plot = y_err[mask]
            if len(x_plot) < 2:
                continue
            color = colors[j % len(colors)]
            ax.errorbar(x_plot, y_plot, yerr=y_err_plot, fmt='o', color=color,
                        ecolor='black', capsize=5, alpha=0.5)
            logx, logy = np.log(x_plot), np.log(y_plot)
            p, cov = np.polyfit(logx, logy, 1, cov=True)
            slope, intercept = p
            slope_error = np.sqrt(cov[0, 0])
            x_fit = np.linspace(np.min(x_plot), np.max(x_plot), 100)
            y_fit = np.exp(intercept) * x_fit**slope
            ax.plot(x_fit, y_fit, '--', color=color,
                    label=f"L={N} fit: {slope:.3f}±{slope_error:.3f}")
            
        ax.set_xscale('log')
        ax.set_yscale('log')
        xlabel = r"Reduced Temperature, $\mathbf{(T_c-T)/T_c}$" if side.lower() == 'lower' else r"Reduced Temperature, $\mathbf{(T_c-T)/T_c}$"
        ax.set_xlabel(xlabel, fontsize=16)
        ax.set_ylabel(obs_titles[obs], fontsize=16)
        ax.legend(fontsize=12,loc="upper right") if i == 2 else ax.legend(fontsize=12)
        
        ax.tick_params(axis='both', which='major', width=3)
        ax.tick_params(axis='both', which='minor', width=1.5)

    
    ax = axs[3]
    sizes = []
    xi_target = []
    xi_errs = []
    for N, data in sorted(observables_data.items()):
        if data["corr"] is None:
            continue
        idx = np.argmin(np.abs(data["t_vals"] - T_target))
        corr = data["corr"][idx]
        corr_err = data["corr_err"][idx] if "corr_err" in data else None
        xi, xi_err = compute_correlation_length(corr, N, corr_err)
        sizes.append(N)
        xi_target.append(xi)
        xi_errs.append(xi_err)
    sizes = np.array(sizes)
    xi_target = np.array(xi_target)
    xi_errs = np.array(xi_errs)
    ax.errorbar(sizes, xi_target, yerr=xi_errs, fmt='o-', capsize=5,
                label=f"T={T_target:.3f}", ecolor='black')
    if len(sizes) > 1:
        p, cov = np.polyfit(np.log(sizes), np.log(xi_target), 1, cov=True)
        slope, intercept = p
        slope_error = np.sqrt(cov[0, 0])
        fit_line = np.exp(intercept) * sizes**slope
        ax.loglog(sizes, fit_line, '--',
                  label=f"Fit: slope = {slope:.2f}±{slope_error:.2f}")
    ax.set_xlabel("System Size, L", fontsize=18)
    ax.set_ylabel("Correlation Length, $\\xi$", fontsize=15)
    ax.legend(fontsize=15)
    ax.tick_params(axis='both', which='major', width=3)
    ax.tick_params(axis='both', which='minor', width=1.5)

    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig("C:/Users/gianc/Documents/isingproject/final figs/exp.pdf", format="pdf", bbox_inches="tight")
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
    plt.tight_layout()
    plt.show()
    

def plot_mag_vs_iterations_comparison(primary_folder, secondary_folder, temperature, system_size, max_iterations=None):

    primary_pattern = os.path.join(primary_folder, f"*T{temperature}*N{system_size}*.npz")
    primary_files = natsort.natsorted(glob.glob(primary_pattern))
    if not primary_files:
        print(f"No file found in {primary_folder} for T={temperature} and N={system_size}")
        return
    primary_file = primary_files[0]
    data_primary = np.load(primary_file)
    mag_primary = np.abs(data_primary['magnetisation'])
    
    secondary_pattern = os.path.join(secondary_folder, f"*T{temperature}*N{system_size}*.npz")
    secondary_files = natsort.natsorted(glob.glob(secondary_pattern))
    if not secondary_files:
        print(f"No file found in {secondary_folder} for T={temperature} and N={system_size}")
        return
    secondary_file = secondary_files[0]
    data_secondary = np.load(secondary_file)
    mag_secondary = np.abs(data_secondary['magnetisation'])
    
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
    plt.ylabel("Magnetisation, M")
    plt.legend(fontsize=14, loc="upper right")
    plt.savefig("C:/Users/gianc/Documents/isingproject/final figs/w_vs_m_final.pdf", format="pdf", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    folder = "final_cluster_exponent"  
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
        
    
    mpl.rcParams.update({
        "figure.dpi": 300,        
        "savefig.dpi": 300,      
        "font.size": 13,
        "axes.titlesize": 15,
        "axes.labelsize": 13,
        "legend.fontsize": 20,
        "text.usetex": False, 
        "font.family": "sans-serif",  
        "font.sans-serif": ["DejaVu Sans"],  
        "font.weight": "bold",
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "mathtext.fontset": "dejavusans",
        "axes.linewidth": 0.8, 
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "lines.linewidth": 1.0,
    })

    
    compute_analysis(input_folder=folder, output_folder=folder, sizes=n_list, dim=2)
    analysis_data = load_analysis(folder=folder, sizes=n_list)
    
    # binder_range_low = 2.26
    # binder_range_high = 2.3
    
    binder_range_low = 2.1
    binder_range_high = 2.5
    
    intersection_range_low = 2.26
    intersection_range_high = 2.27
   
    high_T_threshold = 3.0
    
    # binder_range_low = 4.4
    # binder_range_high = 4.6
    #plot_binder_values(analysis_data, custom_T_low=binder_range_low, custom_T_high=binder_range_high)
    
    
    intersections = calculate_all_intersections(analysis_data, 
                                               custom_T_low=intersection_range_low, 
                                               custom_T_high=intersection_range_high)
    # plot_all_intersections(analysis_data, intersections, zoom_margin=0.005, custom_T_low=2.2, custom_T_high=2.3)
    
    #plot_all_intersections(analysis_data, intersections, zoom_margin=0.005, custom_T_low=2.2685, custom_T_high=2.2695)
    # plot_all_intersections(analysis_data, intersections, zoom_margin=0.005, custom_T_low=4.5, custom_T_high=4.515)
    critical_temps, ref_size = calculate_critical_temperatures(analysis_data, custom_T_low=intersection_range_low, 
    custom_T_high=intersection_range_high )
    
    #plot_critical_temperatures(critical_temps, ref_size)
    
    observables_data = load_observables_extended(input_folder=folder, sizes=n_list)
    
    plot_observables(observables_data)
    
    T_target = 2.26
    #plot_corr_linear(observables_data, T_targets=(2.269, 3))


    plot_correlation_length(observables_data)
    #plot_fss_xi(observables_data, T_target=T_target)

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
        max_iterations=500  
    )
    critical_temps, reference_size = calculate_critical_temperatures(analysis_data, 
                                      custom_T_low=intersection_range_low, 
                                      custom_T_high=intersection_range_high)
    plot_binder_values_with_critical(analysis_data, critical_temps, reference_size, 
                                         custom_T_low=None, custom_T_high=None,
                                         inset_xlim=None, inset_ylim=None,
                                         binder_text="(a)", binder_text_pos=(0.55, 0.90),
                                         inset_text="(b)", inset_text_pos=(0.1, 0.85))
