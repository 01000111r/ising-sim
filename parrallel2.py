#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
from numba import njit, prange
import math
import os
import multiprocessing as mp
from functools import partial

# =============================================================================
# INITIAL STATE: Create flat lattice for Ising model (2D or 3D)
# =============================================================================
def initialstate(N, dim, model):
    """
    Generates a random spin configuration for the Ising model.
    For 2D: returns a flat array of length N*N.
    For 3D: returns a flat array of length N*N*N.
    Spins are ±1.
    """
    if model != 'ising':
        raise ValueError("Only 'ising' model is supported.")
    if dim == 2:
        n_sites = N * N
        return 2 * np.random.randint(0, 2, size=n_sites) - 1
    elif dim == 3:
        n_sites = N * N * N
        return 2 * np.random.randint(0, 2, size=n_sites) - 1
    else:
        raise ValueError("Dimension must be 2 or 3.")


# =============================================================================
# PRECOMPUTE NEIGHBOR INDICES FOR FLAT LATTICES (2D and 3D)
# =============================================================================
def precompute_neighbors(dim, N):
    """
    Precomputes neighbor indices for a flat lattice.
    For 2D: returns an array of shape (N*N, 4) with neighbors:
           +x, -x, +y, -y.
    For 3D: returns an array of shape (N*N*N, 6) with neighbors:
           +x, -x, +y, -y, +z, -z.
    Assumes periodic boundary conditions and N is a power-of-two.
    """
    if dim == 2:
        n_sites = N * N
        neighbors = np.empty((n_sites, 4), dtype=np.int32)
        for i in range(n_sites):
            x = i % N
            y = i // N
            # +x neighbor
            x_plus = (x + 1) & (N - 1)
            neighbors[i, 0] = x_plus + y * N
            # -x neighbor
            x_minus = (x - 1) & (N - 1)
            neighbors[i, 1] = x_minus + y * N
            # +y neighbor
            y_plus = (y + 1) & (N - 1)
            neighbors[i, 2] = x + y_plus * N
            # -y neighbor
            y_minus = (y - 1) & (N - 1)
            neighbors[i, 3] = x + y_minus * N
        return neighbors
    elif dim == 3:
        n_sites = N * N * N
        neighbors = np.empty((n_sites, 6), dtype=np.int32)
        for i in range(n_sites):
            x = i % N
            y = (i // N) % N
            z = i // (N * N)
            # +x
            x_plus = (x + 1) & (N - 1)
            neighbors[i, 0] = x_plus + y * N + z * N * N
            # -x
            x_minus = (x - 1) & (N - 1)
            neighbors[i, 1] = x_minus + y * N + z * N * N
            # +y
            y_plus = (y + 1) & (N - 1)
            neighbors[i, 2] = x + y_plus * N + z * N * N
            # -y
            y_minus = (y - 1) & (N - 1)
            neighbors[i, 3] = x + y_minus * N + z * N * N
            # +z
            z_plus = (z + 1) & (N - 1)
            neighbors[i, 4] = x + y * N + z_plus * N * N
            # -z
            z_minus = (z - 1) & (N - 1)
            neighbors[i, 5] = x + y * N + z_minus * N * N
        return neighbors
    else:
        raise ValueError("Dimension must be 2 or 3.")


# =============================================================================
# CUSTOM XORSHIFT32 RNG (for optimized random numbers)
# =============================================================================
@njit(inline='always')
def xorshift32(rng_state):
    """
    In-place XORSHIFT32 RNG.
    rng_state is a one-element np.uint32 array.
    """
    x = rng_state[0]
    x ^= x << np.uint32(13)
    x ^= x >> np.uint32(17)
    x ^= x << np.uint32(5)
    rng_state[0] = x
    return x

@njit(inline='always')
def rand_float(rng_state):
    """
    Returns a random float in [0, 1) using XORSHIFT32.
    """
    x = xorshift32(rng_state)
    return (x & np.uint32(0xFFFFFFFF)) / 4294967296.0

@njit(inline='always')
def rand_int(rng_state, N):
    """
    Returns a random integer in [0, N) using the custom RNG.
    """
    return int(rand_float(rng_state) * N)


# =============================================================================
# LOCAL METROPOLIS UPDATE FUNCTIONS
# =============================================================================
@njit
def mcmove2d(config, beta, N, model, delta, rng_state, neighbors):
    """
    One sweep of local Metropolis moves on a 2D lattice stored as a flat array.
    """
    n_sites = N * N
    for k in range(n_sites):
        i = rand_int(rng_state, n_sites)
        r = rand_float(rng_state)
        if model == 0:  # ising
            s = config[i]
            nb = 0
            # 4 neighbors for 2D
            for j in range(4):
                nb += config[neighbors[i, j]]
            cost = 2 * s * nb
            if cost < 0 or r < math.exp(-beta * cost):
                config[i] = -s
    return config

@njit
def mcmove3d(config, beta, N, model, delta, rng_state, neighbors):
    """
    One sweep of local Metropolis moves on a 3D lattice stored as a flat array.
    """
    n_sites = N * N * N
    for k in range(n_sites):
        i = rand_int(rng_state, n_sites)
        r = rand_float(rng_state)
        if model == 0:  # ising
            s = config[i]
            nb = 0
            # 6 neighbors for 3D
            for j in range(6):
                nb += config[neighbors[i, j]]
            cost = 2 * s * nb
            if cost < 0 or r < math.exp(-beta * cost):
                config[i] = -s
    return config


# =============================================================================
# WOLFF CLUSTER UPDATE FUNCTIONS
# =============================================================================
@njit
def wolff_update_2d_ising_opt(config, beta, N, rng_state, neighbors):
    """
    Wolff cluster update for the 2D Ising model on a flat array.
    Uses 4 neighbors.
    """
    n_sites = N * N
    p_add = 1.0 - math.exp(-2.0 * beta)
    seed = rand_int(rng_state, n_sites)
    s0 = config[seed]
    config[seed] = -s0  # flip seed immediately
    stack = np.empty(n_sites, dtype=np.int32)
    stack_ptr = 0
    stack[0] = seed
    stack_ptr = 1

    while stack_ptr > 0:
        stack_ptr -= 1
        current = stack[stack_ptr]
        for j in range(4):
            nb_index = neighbors[current, j]
            if config[nb_index] == s0:
                if rand_float(rng_state) < p_add:
                    config[nb_index] = -s0
                    stack[stack_ptr] = nb_index
                    stack_ptr += 1
    return config

@njit
def wolff_update_3d_ising_opt(config, beta, N, rng_state, neighbors):
    """
    Wolff cluster update for the 3D Ising model on a flat array.
    Uses 6 neighbors.
    """
    n_sites = N * N * N
    p_add = 1.0 - math.exp(-2.0 * beta)
    seed = rand_int(rng_state, n_sites)
    s0 = config[seed]
    config[seed] = -s0  # flip seed immediately
    stack = np.empty(n_sites, dtype=np.int32)
    stack_ptr = 0
    stack[0] = seed
    stack_ptr = 1

    while stack_ptr > 0:
        stack_ptr -= 1
        current = stack[stack_ptr]
        for j in range(6):
            nb_index = neighbors[current, j]
            if config[nb_index] == s0:
                if rand_float(rng_state) < p_add:
                    config[nb_index] = -s0
                    stack[stack_ptr] = nb_index
                    stack_ptr += 1
    return config

@njit
def wolff_update_2d(config, beta, N, model, rng_state, neighbors):
    """
    Wrapper for 2D Wolff update.
    """
    if model == 0:
        return wolff_update_2d_ising_opt(config, beta, N, rng_state, neighbors)
    else:
        return config

@njit
def wolff_update_3d(config, beta, N, model, rng_state, neighbors):
    """
    Wrapper for 3D Wolff update.
    """
    if model == 0:
        return wolff_update_3d_ising_opt(config, beta, N, rng_state, neighbors)
    else:
        return config


# =============================================================================
# MEASUREMENT FUNCTIONS
# =============================================================================
@njit
def measure2d_all(config, N, model):
    """
    Measures energy, magnetisation, and spatial correlation along x for a 2D lattice.
    Energy is computed by summing over +x and +y bonds.
    """
    n_sites = N * N
    energy = 0.0
    if model == 0:
        mag = 0.0
        R = N >> 1
        corr = np.zeros(R+1, dtype=np.float64)
        for i in range(n_sites):
            s = config[i]
            x = i % N
            y = i // N
            xp = ((x + 1) & (N - 1)) + y * N
            yp = x + ((y + 1) & (N - 1)) * N
            energy += -s * (config[xp] + config[yp])
            mag += s
            for r in range(R+1):
                x_r = (x + r) & (N - 1)
                j = x_r + y * N
                corr[r] += s * config[j]
        for r in range(R+1):
            corr[r] /= n_sites
        return energy, mag, corr
    else:
        return 0.0, 0.0, np.zeros(1, dtype=np.float64)

@njit
def measure3d_all(config, N, model):
    """
    Measures energy, magnetisation, and spatial correlation along x for a 3D lattice.
    Energy is computed by summing over +x, +y, and +z bonds.
    """
    n_sites = N * N * N
    R = N >> 1
    energy = 0.0
    if model == 0:
        mag = 0.0
        corr = np.zeros(R+1, dtype=np.float64)
        for i in range(n_sites):
            s = config[i]
            x = i % N
            y = (i // N) % N
            z = i // (N * N)
            xp = ((x + 1) & (N - 1)) + y * N + z * N * N
            yp = x + ((y + 1) & (N - 1)) * N + z * N * N
            zp = x + y * N + ((z + 1) & (N - 1)) * N * N
            energy += -s * (config[xp] + config[yp] + config[zp])
            mag += s
            for r in range(R+1):
                x_r = (x + r) & (N - 1)
                j = x_r + y * N + z * N * N
                corr[r] += s * config[j]
        for r in range(R+1):
            corr[r] /= n_sites
        return energy, mag, corr
    else:
        return 0.0, 0.0, np.zeros(R+1, dtype=np.float64)


# =============================================================================
# SIMULATION ROUTINE
# =============================================================================
def run_simulation(config, beta, eqSteps, mcSteps, N, dim, model, delta, update_type):
    """
    Runs equilibration then measurement sweeps.
    Uses flat array storage, precomputed neighbors, and custom RNG.
    Selects update method based on `update_type` ('local' or 'cluster') and dimension.
    """
    mcode = 0  # only 'ising' model supported
    # Set up RNG state and neighbors according to dim.
    if dim == 2:
        n_sites = N * N
    elif dim == 3:
        n_sites = N * N * N
    else:
        raise ValueError("Dimension must be 2 or 3.")
    
    rng_state = np.array([123456789], dtype=np.uint32)
    neighbors = precompute_neighbors(dim, N)
    
    # Equilibration:
    for _ in range(eqSteps):
        if update_type == 'local':
            if dim == 2:
                mcmove2d(config, beta, N, mcode, delta, rng_state, neighbors)
            elif dim == 3:
                mcmove3d(config, beta, N, mcode, delta, rng_state, neighbors)
        elif update_type == 'cluster':
            if dim == 2:
                wolff_update_2d(config, beta, N, mcode, rng_state, neighbors)
            elif dim == 3:
                wolff_update_3d(config, beta, N, mcode, rng_state, neighbors)
    
    # Measurement sweeps:
    m_i = np.empty(mcSteps, dtype=np.float64)
    e_i = np.empty(mcSteps, dtype=np.float64)
    if dim == 2:
        R = N >> 1
    else:
        R = N >> 1
    corr_sum = np.zeros(R+1, dtype=np.float64)
    
    for i in range(mcSteps):
        if update_type == 'local':
            if dim == 2:
                mcmove2d(config, beta, N, mcode, delta, rng_state, neighbors)
            elif dim == 3:
                mcmove3d(config, beta, N, mcode, delta, rng_state, neighbors)
        elif update_type == 'cluster':
            if dim == 2:
                wolff_update_2d(config, beta, N, mcode, rng_state, neighbors)
            elif dim == 3:
                wolff_update_3d(config, beta, N, mcode, rng_state, neighbors)
        if dim == 2:
            e, m, corr = measure2d_all(config, N, mcode)
        elif dim == 3:
            e, m, corr = measure3d_all(config, N, mcode)
        e_i[i] = e
        m_i[i] = m
        corr_sum += corr
    corr_avg = corr_sum / mcSteps
    return m_i, e_i, corr_avg


# =============================================================================
# SIMULATION WRAPPER (Multiprocessing)
# =============================================================================
def simulate_temp(args, output_folder, dim, model, delta, update_type):
    """
    Runs simulation for given N and T_value and saves data.
    """
    N, T_value, eqSteps, mcSteps = args
    beta = 1.0 / T_value
    config = initialstate(N, dim, model)
    m_i, e_i, corr_avg = run_simulation(config, beta, eqSteps, mcSteps, N, dim, model, delta, update_type)
    filename = os.path.join(output_folder, f"run-T{T_value:.3f}N{N}D{dim}-{model}-{update_type}.npz")
    np.savez(filename, energy=e_i, magnetisation=m_i, correlation=corr_avg)
    print(f"Finished: T={T_value:.3f}, N={N}, dim={dim}, model={model}, update={update_type}, avg m/site={m_i.mean()/(N**dim):.3f}")
    return T_value, N, m_i, e_i, corr_avg


# =============================================================================
# DATA CREATION LOOP
# =============================================================================
def create_data(output_folder, nt, n_list, eqSteps, mcSteps, T_arr, dim, model, delta, update_type):
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
             dim=dim,
             model=model,
             delta=delta,
             update_type=update_type)
    print(f"Simulation parameters saved to {params_filepath}")
    tasks = []
    for N in n_list:
        for T_val in T_arr:
            tasks.append((N, T_val, eqSteps, mcSteps))
    total_tasks = len(tasks)
    completed_tasks = 0
    pool = mp.Pool(processes=5)
    sim_func = partial(simulate_temp, output_folder=output_folder, dim=dim, model=model, delta=delta, update_type=update_type)
    for result in pool.imap_unordered(sim_func, tasks):
        completed_tasks += 1
        print(f"Progress: {completed_tasks}/{total_tasks} simulations completed.")
    pool.close()
    pool.join()


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == '__main__':
    output_folder = "data_optimized_ising_3"
    nt          = 50                     # Number of temperature points
    n_list      = [8, 16, 32]     # Lattice sizes (must be power-of-two)
    eqSteps     = 1024 * 1000          # Equilibration sweeps
    mcSteps     = 1024 * 10              # Measurement sweeps
    T_arr       = np.linspace(0, 4, nt)  # Temperature array (adjust as needed)
    # Set dim to 2 or 3 as required.
    dim         = 2                      # 2D simulation; change to 3 for 3D.
    model       = 'ising'                # Ising model
    delta       = 0.3                    # For local moves (not used in cluster updates)
    update_type = 'cluster'                # Choose 'local' for Metropolis or 'cluster' for Wolff
    
    create_data(output_folder, nt, n_list, eqSteps, mcSteps, T_arr, dim, model, delta, update_type)
