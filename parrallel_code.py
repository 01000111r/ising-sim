#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 14:55:37 2025

@author: giancarloramirez
"""

from __future__ import division
import numpy as np
from numba import njit
import math
import os
import multiprocessing as mp
from functools import partial

#----------------------------------------------------------------------
##  CORE FUNCTIONS FOR THE SIMULATION
#----------------------------------------------------------------------

def initialstate(N):
    """
    Generates a random spin configuration for the initial condition.
    Spins are ±1.
    """
    return 2 * np.random.randint(0, 2, size=(N, N)) - 1

@njit
def mcmove(config, beta, N):
    """
    Performs one Monte Carlo sweep (N*N attempted moves)
    using the Metropolis algorithm. Pre-generates random numbers
    for all moves in the sweep.
    """
    num_moves = N * N
    # Pre-generate random numbers for positions and acceptance:
    r1 = np.empty(num_moves, dtype=np.int64)
    r2 = np.empty(num_moves, dtype=np.int64)
    r3 = np.empty(num_moves, dtype=np.float64)
    for k in range(num_moves):
        r1[k] = np.random.randint(0, N)
        r2[k] = np.random.randint(0, N)
        r3[k] = np.random.random()
    
    for idx in range(num_moves):
        a = r1[idx]
        b = r2[idx]
        s = config[a, b]
        # Sum over the four nearest neighbors with periodic boundaries:
        nb = config[(a + 1) % N, b] + config[a, (b + 1) % N] + \
             config[(a - 1) % N, b] + config[a, (b - 1) % N]
        cost = 2 * s * nb
        if cost < 0 or r3[idx] < math.exp(-cost * beta):
            config[a, b] = -s
    return config

@njit
def calcEnergy(config, N):
    """
    Calculates the energy of the configuration.
    This version sums over each spin interacting with its right and down neighbor
    (with periodic boundaries) so that each bond is counted only once.
    """
    energy = 0
    for i in range(N):
        for j in range(N):
            S = config[i, j]
            nb = config[(i + 1) % N, j] + config[i, (j + 1) % N] + \
                 config[(i - 1) % N, j] + config[i, (j - 1) % N]
            energy += -nb * S
            energy -= config[i, j] * (config[(i + 1) % N, j] + config[i, (j + 1) % N])
    return energy / 4

@njit
def calcMag(config):
    """
    Returns the magnetization (sum of all spins).
    """
    return np.sum(config)

@njit
def run_simulation(config, beta, eqSteps, mcSteps, N):
    """
    Consolidated simulation loop that runs equilibration followed by measurement.
    Both loops are JIT–compiled, which minimizes Python–overhead.
    
    Parameters:
      config  : 2D numpy array, the spin configuration.
      beta    : inverse temperature.
      eqSteps : number of equilibration sweeps.
      mcSteps : number of measurement sweeps.
      N       : lattice size.
      
    Returns:
      m_i : 1D numpy array of magnetization measurements.
      e_i : 1D numpy array of energy measurements.
    """
    # Equilibration phase
    for _ in range(eqSteps):
        mcmove(config, beta, N)
        
    # Allocate arrays for measurements
    m_i = np.empty(mcSteps, dtype=np.float64)
    e_i = np.empty(mcSteps, dtype=np.float64)
    
    # Measurement phase
    for i in range(mcSteps):
        mcmove(config, beta, N)
        e_i[i] = calcEnergy(config, N)
        m_i[i] = calcMag(config)
        
    return m_i, e_i

#----------------------------------------------------------------------
##  SIMULATION FUNCTION (to be run in parallel)
#----------------------------------------------------------------------

def simulate_temp(args, output_folder):
    """
    Runs the simulation for a given lattice size and temperature.
    Sets up the initial configuration, calls the consolidated simulation function,
    computes susceptibility and specific heat from the measurement data,
    and saves the resulting data to file.
    
    Parameters:
      args : tuple (N, T_value, eqSteps, mcSteps)
      output_folder : folder where the simulation data will be saved.
    """
    N, T_value, eqSteps, mcSteps = args
    beta = 1.0 / T_value
    config = initialstate(N)
    
    # Run the simulation using the consolidated, JIT-compiled function.
    m_i, e_i = run_simulation(config, beta, eqSteps, mcSteps, N)
    
    # Compute averages and fluctuations.
    m_mean = np.mean(m_i)
    m2_mean = np.mean(m_i**2)
    e_mean = np.mean(e_i)
    e2_mean = np.mean(e_i**2)
    
    # Compute susceptibility and specific heat using the fluctuation formulas.
    susceptibility = (m2_mean - m_mean**2) * beta
    specific_heat   = (e2_mean - e_mean**2) * beta**2
    
    # Save the results to file.
    filename = os.path.join(output_folder, f"run-T{T_value:.3f}N{N}.npz")
    np.savez(filename,
             energy=e_i,
             magnetisation=m_i,
             susceptibility=susceptibility,
             specific_heat=specific_heat)
    
    print(f"Finished: T={T_value:.3f}, N={N}, avg m/site={m_i.mean()/(N*N):.3f}")
    return T_value, N, m_i, e_i, susceptibility, specific_heat

#----------------------------------------------------------------------
##  MAIN FUNCTION TO RUN ALL SIMULATIONS (in parallel)
#----------------------------------------------------------------------

def create_data(output_folder, nt, n_list, eqSteps, mcSteps, T_arr):
    """
    Loops over all lattice sizes and temperatures, launching a simulation
    for each case in parallel using a multiprocessing Pool.
    
    The simulation parameters are saved in the specified output folder.
    """
    # Create output directory if it doesn't exist.
    if os.path.exists(output_folder):
        print(f"Folder '{output_folder}' already exists.")
    else:
        os.makedirs(output_folder)
    
    # Save simulation parameters to an NPZ file in the output folder.
    params_filepath = os.path.join(output_folder, "simulation_parameters.npz")
    np.savez(params_filepath,
             nt=nt,
             n_list=np.array(n_list),
             eqSteps=eqSteps,
             mcSteps=mcSteps,
             T_arr=T_arr)
    print(f"Simulation parameters saved to {params_filepath}")
    
    # Prepare a list of simulation tasks for each (N, T) combination.
    tasks = []
    for N in n_list:
        for T_val in T_arr:
            tasks.append((N, T_val, eqSteps, mcSteps))
    
    total_tasks = len(tasks)
    completed_tasks = 0
    
    # Create a pool of workers equal to the number of available CPU cores.
    pool = mp.Pool(processes=6)
    
    # Use functools.partial to inject the output_folder argument into simulate_temp.
    sim_func = partial(simulate_temp, output_folder=output_folder)
    
    # Use imap_unordered to process tasks as soon as they are finished.
    for result in pool.imap_unordered(sim_func, tasks):
        completed_tasks += 1
        print(f"Progress: {completed_tasks}/{total_tasks} simulations completed.")
    
    pool.close()
    pool.join()

#----------------------------------------------------------------------
##  MAIN EXECUTION: Specify Parameters and Run the Simulation
#----------------------------------------------------------------------

if __name__ == '__main__':
    # *** Set the output folder name and simulation parameters here ***
    output_folder = "data9"       # Folder name where data will be saved
    nt          = 100           # Number of temperature points
    n_list      = [16, 32, 64]       # List of lattice sizes (N x N)
    eqSteps     = 1024*1000    # Number of Monte Carlo sweeps for equilibration
    mcSteps     = 1024*1000          # Number of Monte Carlo sweeps for measurements
    T_arr       = np.linspace(1.75, 2.75, nt)  # Temperature array
    
    # Run the simulation with the specified parameters.
    create_data(output_folder, nt, n_list, eqSteps, mcSteps, T_arr)
