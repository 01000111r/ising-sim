#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from __future__ import division
import numpy as np
from numba import njit
import math
import os
import multiprocessing as mp
from functools import partial

# state config

def initialstate(N, dim):
    """
    Generates a random spin configuration for the initial condition.
    Spins are ±1.
    
    Parameters:
      N   : Lattice size (for 2D: shape (N,N); for 3D: shape (N,N,N))
      dim : Dimension (2 or 3)
    """
    if dim == 2:
        return 2 * np.random.randint(0, 2, size=(N, N)) - 1
    elif dim == 3:
        return 2 * np.random.randint(0, 2, size=(N, N, N)) - 1
    else:
        raise ValueError("Dimension must be 2 or 3.")

# monte carlo moves

@njit
def mcmove2d(config, beta, N):
    """
    Performs one Monte Carlo sweep for a 2D configuration.
    """
    num_moves = N * N
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
        # Sum over four nearest neighbors with periodic boundaries.
        nb = (config[(a + 1) % N, b] + config[a, (b + 1) % N] +
              config[(a - 1) % N, b] + config[a, (b - 1) % N])
        cost = 2 * s * nb
        if cost < 0 or r3[idx] < math.exp(-cost * beta):
            config[a, b] = -s
    return config

@njit
def mcmove3d(config, beta, N):
    """
    Performs one Monte Carlo sweep for a 3D configuration.
    """
    num_moves = N * N * N
    r1 = np.empty(num_moves, dtype=np.int64)
    r2 = np.empty(num_moves, dtype=np.int64)
    r3 = np.empty(num_moves, dtype=np.int64)
    r4 = np.empty(num_moves, dtype=np.float64)
    for k in range(num_moves):
        r1[k] = np.random.randint(0, N)
        r2[k] = np.random.randint(0, N)
        r3[k] = np.random.randint(0, N)
        r4[k] = np.random.random()
    
    for idx in range(num_moves):
        x = r1[idx]
        y = r2[idx]
        z = r3[idx]
        s = config[x, y, z]
        # Sum over the six nearest neighbors in 3D (with periodic boundaries).
        nb = (config[(x+1)%N, y, z] + config[(x-1)%N, y, z] +
              config[x, (y+1)%N, z] + config[x, (y-1)%N, z] +
              config[x, y, (z+1)%N] + config[x, y, (z-1)%N])
        cost = 2 * s * nb
        if cost < 0 or r4[idx] < math.exp(-cost * beta):
            config[x, y, z] = -s
    return config


# Energy calc


@njit
def calcEnergy2d(config, N):
    """
    Calculates the energy for a 2D configuration.
    Only right and down neighbors are summed to avoid double counting.
    """
    energy = 0
    for i in range(N):
        for j in range(N):
            S = config[i, j]
            energy += -S * (config[i, (j+1)%N] + config[(i+1)%N, j])
    return energy

@njit
def calcEnergy3d(config, N):
    """
    Calculates the energy for a 3D configuration.
    Only neighbors in the positive directions are summed to avoid double counting.
    """
    energy = 0
    for x in range(N):
        for y in range(N):
            for z in range(N):
                S = config[x, y, z]
                energy += -S * (config[(x+1)%N, y, z] +
                                config[x, (y+1)%N, z] +
                                config[x, y, (z+1)%N])
    return energy

# Magnetisation calc

@njit
def calcMag2d(config):
    """
    Returns the magnetisation (sum of all spins) for a 2D configuration.
    """
    return np.sum(config)

@njit
def calcMag3d(config, N):
    """
    Returns the magnetisation (sum of all spins) for a 3D configuration.
    """
    mag = 0
    for x in range(N):
        for y in range(N):
            for z in range(N):
                mag += config[x, y, z]
    return mag


# running code


def run_simulation(config, beta, eqSteps, mcSteps, N, dim):
    """
    Runs equilibration followed by measurement sweeps.
    Selects the appropriate functions based on the dimension.
    
    Parameters:
      config  : initial spin configuration.
      beta    : inverse temperature.
      eqSteps : number of equilibration sweeps.
      mcSteps : number of measurement sweeps.
      N       : lattice size.
      dim     : simulation dimension (2 or 3).
      
    Returns:
      m_i : 1D numpy array of magnetisation measurements.
      e_i : 1D numpy array of energy measurements.
    """
    if dim == 2:
        for _ in range(eqSteps):
            mcmove2d(config, beta, N)
        m_i = np.empty(mcSteps, dtype=np.float64)
        e_i = np.empty(mcSteps, dtype=np.float64)
        for i in range(mcSteps):
            mcmove2d(config, beta, N)
            e_i[i] = calcEnergy2d(config, N)
            m_i[i] = calcMag2d(config)
        return m_i, e_i
    elif dim == 3:
        for _ in range(eqSteps):
            mcmove3d(config, beta, N)
        m_i = np.empty(mcSteps, dtype=np.float64)
        e_i = np.empty(mcSteps, dtype=np.float64)
        for i in range(mcSteps):
            mcmove3d(config, beta, N)
            e_i[i] = calcEnergy3d(config, N)
            m_i[i] = calcMag3d(config, N)
        return m_i, e_i
    else:
        raise ValueError("Dimension must be 2 or 3.")

def simulate_temp(args, output_folder, dim):
    """
    Runs the simulation for a given lattice size and temperature, then saves the
    raw measurement data. The output file name is modified to include the dimension.
    
    Parameters:
      args          : tuple (N, T_value, eqSteps, mcSteps)
      output_folder : folder to save the simulation data.
      dim           : simulation dimension (2 or 3).
    """
    N, T_value, eqSteps, mcSteps = args
    beta = 1.0 / T_value
    config = initialstate(N, dim)
    

    m_i, e_i = run_simulation(config, beta, eqSteps, mcSteps, N, dim)
    

    filename = os.path.join(output_folder, f"run-T{T_value:.3f}N{N}D{dim}.npz")
    np.savez(filename,
             energy=e_i,
             magnetisation=m_i)
    

    print(f"Finished: T={T_value:.3f}, N={N}, dim={dim}, avg m/site={m_i.mean()/(N**dim):.3f}")
    return T_value, N, m_i, e_i



def create_data(output_folder, nt, n_list, eqSteps, mcSteps, T_arr, dim):
    """
    Loops over all lattice sizes and temperatures, launching a simulation
    for each case in parallel. The simulation parameters (including dimension)
    are saved to the output folder.
    
    Parameters:
      output_folder : folder where data and parameters are saved.
      nt            : number of temperature points.
      n_list        : list of lattice sizes.
      eqSteps       : number of equilibration sweeps.
      mcSteps       : number of measurement sweeps.
      T_arr         : array of temperatures.
      dim           : simulation dimension (2 or 3).
    """
    if os.path.exists(output_folder):
        print(f"Folder '{output_folder}' already exists.")
    else:
        os.makedirs(output_folder)
    
    params_filepath = os.path.join(output_folder, "simulation_parameters.npz")
    np.savez(params_filepath,
             nt=nt,
             n_list=np.array(n_list),
             eqSteps=eqSteps,
             mcSteps=mcSteps,
             T_arr=T_arr,
             dim=dim)
    print(f"Simulation parameters saved to {params_filepath}")
    
    tasks = []
    for N in n_list:
        for T_val in T_arr:
            tasks.append((N, T_val, eqSteps, mcSteps))
    
    total_tasks = len(tasks)
    completed_tasks = 0
    
    pool = mp.Pool(processes=5)
    sim_func = partial(simulate_temp, output_folder=output_folder, dim=dim)
    
    for result in pool.imap_unordered(sim_func, tasks):
        completed_tasks += 1
        print(f"Progress: {completed_tasks}/{total_tasks} simulations completed.")
    
    pool.close()
    pool.join()


if __name__ == '__main__':

    output_folder = "data10"       
    nt          = 50             # Number of temperature points
    n_list      = [16, 32, 64]    # List of lattice sizes 
    eqSteps     = 1024 * 10     # Number of equilibration sweeps
    mcSteps     = 1024 * 10    # Number of measurement sweeps
    T_arr       = np.linspace(3.5, 5.5, nt)  # Temperature array
    dim         = 3              
    
    create_data(output_folder, nt, n_list, eqSteps, mcSteps, T_arr, dim)
