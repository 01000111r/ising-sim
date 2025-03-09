#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimized 3D Ising simulation with:
  1. Flat array storage and precomputed neighbor indices.
  2. An inline Xorshift RNG for fast random number generation (fixed division).
  3. GPU-accelerated measurement of energy and magnetization.
  
This version moves the multiprocessing worker function to global scope.
"""

from __future__ import division
import numpy as np
from numba import njit, cuda, prange
import math
import os
import multiprocessing as mp

# -----------------------------------------------------------------------------
# Global: Precompute neighbor indices for a flat 3D lattice.
# For periodic boundaries, we assume N is a power of 2 so that modulo N is replaced
# by bitwise & (N-1). For energy, we only need the "forward" neighbors in x, y, z.
# For the Wolff update we precompute all 6 neighbors.
# -----------------------------------------------------------------------------
def precompute_neighbors(N):
    N3 = N * N * N
    neigh_all = np.empty((N3, 6), dtype=np.int32)  # for Wolff update: 6 neighbors
    neigh_meas = np.empty((N3, 3), dtype=np.int32)  # for energy measurement: +x, +y, +z
    for z in range(N):
        for y in range(N):
            for x in range(N):
                i = x + N * (y + N * z)
                xp = (x + 1) & (N - 1)
                xm = (x - 1) & (N - 1)
                yp = (y + 1) & (N - 1)
                ym = (y - 1) & (N - 1)
                zp = (z + 1) & (N - 1)
                zm = (z - 1) & (N - 1)
                neigh_all[i, 0] = xp + N * (y + N * z)
                neigh_all[i, 1] = xm + N * (y + N * z)
                neigh_all[i, 2] = x + N * (yp + N * z)
                neigh_all[i, 3] = x + N * (ym + N * z)
                neigh_all[i, 4] = x + N * (y + N * zp)
                neigh_all[i, 5] = x + N * (y + N * zm)
                neigh_meas[i, 0] = neigh_all[i, 0]
                neigh_meas[i, 1] = neigh_all[i, 2]
                neigh_meas[i, 2] = neigh_all[i, 4]
    return neigh_all, neigh_meas

# -----------------------------------------------------------------------------
# Initial State: Create a flat array configuration.
# -----------------------------------------------------------------------------
def initialstate(N, dim, model):
    """
    Generates a random spin configuration stored as a flat array.
    For 'ising': spins are ±1.
    """
    if model == 'ising':
        N3 = N * N * N
        return 2 * np.random.randint(0, 2, size=N3).astype(np.int8) - 1
    else:
        raise ValueError("Model must be 'ising', 'xy', or 'heisenberg'.")

# -----------------------------------------------------------------------------
# Xorshift RNG functions (32-bit) with fixed types.
# -----------------------------------------------------------------------------
@njit(inline='always')
def xorshift32(state):
    mask = np.uint32(0xFFFFFFFF)
    state ^= (state << np.uint32(13)) & mask
    state ^= (state >> np.uint32(17))
    state ^= (state << np.uint32(5)) & mask
    return state & mask

@njit(inline='always')
def rand_uniform(state):
    state = xorshift32(state)
    # Use a float constant for denominator so that we don't get zero.
    return state / 4294967296.0, state

# -----------------------------------------------------------------------------
# Wolff Cluster Update (flat-array version using precomputed neighbors and Xorshift RNG)
# -----------------------------------------------------------------------------
@njit
def wolff_update_3d_ising_flat(config, beta, N, neigh_all):
    N3 = N * N * N
    p_add = 1.0 - math.exp(-2.0 * beta)
    state = np.uint32(123456789)  # initial RNG state
    seed_index = np.random.randint(0, N3)
    s0 = config[seed_index]
    config[seed_index] = -s0  # flip seed to mark as visited
    stack = np.empty(N3, dtype=np.int32)
    stack_ptr = 0
    stack[0] = seed_index
    stack_ptr = 1

    while stack_ptr > 0:
        stack_ptr -= 1
        i = stack[stack_ptr]
        for nb in neigh_all[i]:
            if config[nb] == s0:
                r, state = rand_uniform(state)
                if r < p_add:
                    config[nb] = -s0
                    stack[stack_ptr] = nb
                    stack_ptr += 1
    return config

@njit
def wolff_update_3d(config, beta, N, model, neigh_all):
    if model == 0:
        return wolff_update_3d_ising_flat(config, beta, N, neigh_all)
    else:
        return config

# -----------------------------------------------------------------------------
# GPU Kernel for measurement (energy and magnetization)
# -----------------------------------------------------------------------------
@cuda.jit
def measure_kernel(config, neigh_meas, energy_global, mag_global, N3):
    i = cuda.grid(1)
    if i < N3:
        S = config[i]
        e = -S * (config[neigh_meas[i, 0]] +
                  config[neigh_meas[i, 1]] +
                  config[neigh_meas[i, 2]])
        cuda.atomic.add(energy_global, 0, e)
        cuda.atomic.add(mag_global, 0, S)

def measure3d_gpu(config, neigh_meas, N):
    N3 = N * N * N
    d_config = cuda.to_device(config)
    d_neigh_meas = cuda.to_device(neigh_meas)
    energy_global = np.zeros(1, dtype=np.float64)
    mag_global = np.zeros(1, dtype=np.float64)
    d_energy = cuda.to_device(energy_global)
    d_mag = cuda.to_device(mag_global)
    
    threadsperblock = 256
    blockspergrid = (N3 + (threadsperblock - 1)) // threadsperblock
    measure_kernel[blockspergrid, threadsperblock](d_config, d_neigh_meas, d_energy, d_mag, N3)
    d_energy.copy_to_host(energy_global)
    d_mag.copy_to_host(mag_global)
    return energy_global[0], mag_global[0]

# -----------------------------------------------------------------------------
# Simulation Routine (using flat array, our Wolff update and GPU measurement)
# -----------------------------------------------------------------------------
def run_simulation(config, beta, eqSteps, mcSteps, N, dim, model, update_type, neigh_all, neigh_meas):
    mcode = 0  # only 'ising' is implemented
    for _ in range(eqSteps):
        if update_type == 'cluster':
            config = wolff_update_3d(config, beta, N, mcode, neigh_all)
    m_i = np.empty(mcSteps, dtype=np.float64)
    e_i = np.empty(mcSteps, dtype=np.float64)
    for i in range(mcSteps):
        if update_type == 'cluster':
            config = wolff_update_3d(config, beta, N, mcode, neigh_all)
        e, m = measure3d_gpu(config, neigh_meas, N)
        e_i[i] = e
        m_i[i] = m
    return m_i, e_i

# -----------------------------------------------------------------------------
# Simulation Wrapper (for multiprocessing)
# -----------------------------------------------------------------------------
def simulate_temp(args):
    # Global worker function (picklable)
    N, T_val, eqSteps, mcSteps, neigh_all, neigh_meas, output_folder, dim, model, update_type = args
    beta = 1.0 / T_val
    config = initialstate(N, dim, model)
    m_i, e_i = run_simulation(config, beta, eqSteps, mcSteps, N, dim, model, update_type, neigh_all, neigh_meas)
    filename = os.path.join(output_folder, f"run-T{T_val:.3f}N{N}D{dim}-{model}-{update_type}.npz")
    np.savez(filename, energy=e_i, magnetisation=m_i)
    print(f"Finished: T={T_val:.3f}, N={N}, dim={dim}, model={model}, update={update_type}, avg m/site={m_i.mean()/(N**3):.3f}")
    return T_val, N, m_i, e_i

# -----------------------------------------------------------------------------
# Data Creation Loop
# -----------------------------------------------------------------------------
def create_data(output_folder, nt, n_list, eqSteps, mcSteps, T_arr, dim, model, update_type):
    if os.path.exists(output_folder):
        print(f"Folder '{output_folder}' already exists.")
    else:
        os.makedirs(output_folder)
    params_filepath = os.path.join(output_folder, "simulation_parameters.npz")
    np.savez(params_filepath,
             nt=nt, n_list=np.array(n_list),
             eqSteps=eqSteps, mcSteps=mcSteps,
             T_arr=T_arr, dim=dim, model=model,
             update_type=update_type)
    print(f"Simulation parameters saved to {params_filepath}")
    tasks = []
    for N in n_list:
        neigh_all, neigh_meas = precompute_neighbors(N)
        for T_val in T_arr:
            tasks.append((N, T_val, eqSteps, mcSteps, neigh_all, neigh_meas,
                          output_folder, dim, model, update_type))
    total_tasks = len(tasks)
    completed_tasks = 0
    pool = mp.Pool(processes=5)
    for result in pool.imap_unordered(simulate_temp, tasks):
        completed_tasks += 1
        print(f"Progress: {completed_tasks}/{total_tasks} simulations completed.")
    pool.close()
    pool.join()

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    output_folder = "data17_3d_ising_optimized_flat_gpu"
    nt          = 50                    # Number of temperature points
    n_list      = [32, 64, 128, 256]          # Lattice sizes
    eqSteps     = 1024 * 10              # Equilibration sweeps (adjust as needed)
    mcSteps     = 1024 * 5               # Measurement sweeps (adjust as needed)
    T_arr       = np.linspace(0, 8, nt)  # Temperature array
    dim         = 3                      # 2 or 3
    model       = 'ising'                # 'ising', 'xy', or 'heisenberg'
    update_type = 'cluster'              # 'local' or 'cluster'
    
    create_data(output_folder, nt, n_list, eqSteps, mcSteps, T_arr, dim, model, update_type)
