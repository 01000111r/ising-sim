#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
from numba import njit
import math
import os
import multiprocessing as mp
from functools import partial

# Initialise state

def initialstate(N, dim, model):
    """
    Generates a random spin configuration.
    
    For 'ising': spins are ±1.
    For 'xy'   : spins are angles in [0,2π).
    For 'heisenberg': spins are 3D unit vectors.
    
    For 2D: shape (N,N) or (N,N,3); for 3D: shape (N,N,N) or (N,N,N,3).
    """
    if model == 'ising':
        if dim == 2:
            return 2 * np.random.randint(0, 2, size=(N, N)) - 1
        elif dim == 3:
            return 2 * np.random.randint(0, 2, size=(N, N, N)) - 1
    elif model == 'xy':
        if dim == 2:
            return np.random.uniform(0, 2*math.pi, size=(N, N))
        elif dim == 3:
            return np.random.uniform(0, 2*math.pi, size=(N, N, N))
    elif model == 'heisenberg':
        if dim == 2:
            config = np.random.normal(size=(N, N, 3))
            for i in range(N):
                for j in range(N):
                    norm = math.sqrt(config[i, j, 0]**2 + config[i, j, 1]**2 + config[i, j, 2]**2)
                    config[i, j, 0] /= norm
                    config[i, j, 1] /= norm
                    config[i, j, 2] /= norm
            return config
        elif dim == 3:
            config = np.random.normal(size=(N, N, N, 3))
            for i in range(N):
                for j in range(N):
                    for k in range(N):
                        norm = math.sqrt(config[i, j, k, 0]**2 + config[i, j, k, 1]**2 + config[i, j, k, 2]**2)
                        config[i, j, k, 0] /= norm
                        config[i, j, k, 1] /= norm
                        config[i, j, k, 2] /= norm
            return config
    else:
        raise ValueError("Model must be 'ising', 'xy', or 'heisenberg'.")

# Metropolis Algorithms

@njit
def mcmove2d(config, beta, N, model, delta):
    """
    One sweep of local Metropolis moves on a 2D lattice.
    model: 0=ising, 1=xy, 2=heisenberg.
    """
    num_moves = N * N
    for k in range(num_moves):
        i = np.random.randint(0, N)
        j = np.random.randint(0, N)
        r = np.random.random()
        if model == 0:
            s = config[i, j]
            nb = (config[(i+1)%N, j] + config[(i-1)%N, j] +
                  config[i, (j+1)%N] + config[i, (j-1)%N])
            cost = 2 * s * nb
            if cost < 0 or r < math.exp(-beta * cost):
                config[i, j] = -s
        elif model == 1:
            theta = config[i, j]
            theta_r = config[(i+1)%N, j]
            theta_l = config[(i-1)%N, j]
            theta_u = config[i, (j+1)%N]
            theta_d = config[i, (j-1)%N]
            oldE = - (math.cos(theta - theta_r) + math.cos(theta - theta_l) +
                      math.cos(theta - theta_u) + math.cos(theta - theta_d))
            new_theta = theta + np.random.uniform(-delta, delta)
            newE = - (math.cos(new_theta - theta_r) + math.cos(new_theta - theta_l) +
                      math.cos(new_theta - theta_u) + math.cos(new_theta - theta_d))
            dE = newE - oldE
            if dE < 0 or r < math.exp(-beta * dE):
                config[i, j] = new_theta
        elif model == 2:
            s0 = config[i, j, 0]
            s1 = config[i, j, 1]
            s2 = config[i, j, 2]
            nb0 = (config[(i+1)%N, j, 0] + config[(i-1)%N, j, 0] +
                   config[i, (j+1)%N, 0] + config[i, (j-1)%N, 0])
            nb1 = (config[(i+1)%N, j, 1] + config[(i-1)%N, j, 1] +
                   config[i, (j+1)%N, 1] + config[i, (j-1)%N, 1])
            nb2 = (config[(i+1)%N, j, 2] + config[(i-1)%N, j, 2] +
                   config[i, (j+1)%N, 2] + config[i, (j-1)%N, 2])
            oldE = - (s0 * nb0 + s1 * nb1 + s2 * nb2)
            new0 = s0 + delta * np.random.normal()
            new1 = s1 + delta * np.random.normal()
            new2 = s2 + delta * np.random.normal()
            norm = math.sqrt(new0*new0 + new1*new1 + new2*new2)
            new0 /= norm; new1 /= norm; new2 /= norm
            new_nb0 = (config[(i+1)%N, j, 0] + config[(i-1)%N, j, 0] +
                       config[i, (j+1)%N, 0] + config[i, (j-1)%N, 0])
            new_nb1 = (config[(i+1)%N, j, 1] + config[(i-1)%N, j, 1] +
                       config[i, (j+1)%N, 1] + config[i, (j-1)%N, 1])
            new_nb2 = (config[(i+1)%N, j, 2] + config[(i-1)%N, j, 2] +
                       config[i, (j+1)%N, 2] + config[i, (j-1)%N, 2])
            newE = - (new0 * new_nb0 + new1 * new_nb1 + new2 * new_nb2)
            dE = newE - oldE
            if dE < 0 or r < math.exp(-beta * dE):
                config[i, j, 0] = new0
                config[i, j, 1] = new1
                config[i, j, 2] = new2
    return config

@njit
def mcmove3d(config, beta, N, model, delta):
    """
    One sweep of local Metropolis moves on a 3D lattice.
    """
    num_moves = N * N * N
    for k in range(num_moves):
        x = np.random.randint(0, N)
        y = np.random.randint(0, N)
        z = np.random.randint(0, N)
        r = np.random.random()
        if model == 0:
            s = config[x, y, z]
            nb = (config[(x+1)%N, y, z] + config[(x-1)%N, y, z] +
                  config[x, (y+1)%N, z] + config[x, (y-1)%N, z] +
                  config[x, y, (z+1)%N] + config[x, y, (z-1)%N])
            cost = 2 * s * nb
            if cost < 0 or r < math.exp(-beta * cost):
                config[x, y, z] = -s
        elif model == 1:
            theta = config[x, y, z]
            t1 = config[(x+1)%N, y, z]
            t2 = config[(x-1)%N, y, z]
            t3 = config[x, (y+1)%N, z]
            t4 = config[x, (y-1)%N, z]
            t5 = config[x, y, (z+1)%N]
            t6 = config[x, y, (z-1)%N]
            oldE = - (math.cos(theta - t1) + math.cos(theta - t2) +
                      math.cos(theta - t3) + math.cos(theta - t4) +
                      math.cos(theta - t5) + math.cos(theta - t6))
            new_theta = theta + np.random.uniform(-delta, delta)
            newE = - (math.cos(new_theta - t1) + math.cos(new_theta - t2) +
                      math.cos(new_theta - t3) + math.cos(new_theta - t4) +
                      math.cos(new_theta - t5) + math.cos(new_theta - t6))
            dE = newE - oldE
            if dE < 0 or r < math.exp(-beta * dE):
                config[x, y, z] = new_theta
        elif model == 2:
            s0 = config[x, y, z, 0]
            s1 = config[x, y, z, 1]
            s2 = config[x, y, z, 2]
            nb0 = (config[(x+1)%N, y, z, 0] + config[(x-1)%N, y, z, 0] +
                   config[x, (y+1)%N, z, 0] + config[x, (y-1)%N, z, 0] +
                   config[x, y, (z+1)%N, 0] + config[x, y, (z-1)%N, 0])
            nb1 = (config[(x+1)%N, y, z, 1] + config[(x-1)%N, y, z, 1] +
                   config[x, (y+1)%N, z, 1] + config[x, (y-1)%N, z, 1] +
                   config[x, y, (z+1)%N, 1] + config[x, y, (z-1)%N, 1])
            nb2 = (config[(x+1)%N, y, z, 2] + config[(x-1)%N, y, z, 2] +
                   config[x, (y+1)%N, z, 2] + config[x, (y-1)%N, z, 2] +
                   config[x, y, (z+1)%N, 2] + config[x, y, (z-1)%N, 2])
            oldE = - (s0 * nb0 + s1 * nb1 + s2 * nb2)
            new0 = s0 + delta * np.random.normal()
            new1 = s1 + delta * np.random.normal()
            new2 = s2 + delta * np.random.normal()
            norm = math.sqrt(new0*new0 + new1*new1 + new2*new2)
            new0 /= norm; new1 /= norm; new2 /= norm
            new_nb0 = (config[(x+1)%N, y, z, 0] + config[(x-1)%N, y, z, 0] +
                       config[x, (y+1)%N, z, 0] + config[x, (y-1)%N, z, 0] +
                       config[x, y, (z+1)%N, 0] + config[x, y, (z-1)%N, 0])
            new_nb1 = (config[(x+1)%N, y, z, 1] + config[(x-1)%N, y, z, 1] +
                       config[x, (y+1)%N, z, 1] + config[x, (y-1)%N, z, 1] +
                       config[x, y, (z+1)%N, 1] + config[x, y, (z-1)%N, 1])
            new_nb2 = (config[(x+1)%N, y, z, 2] + config[(x-1)%N, y, z, 2] +
                       config[x, (y+1)%N, z, 2] + config[x, (y-1)%N, z, 2] +
                       config[x, y, (z+1)%N, 2] + config[x, y, (z-1)%N, 2])
            newE = - (new0 * new_nb0 + new1 * new_nb1 + new2 * new_nb2)
            dE = newE - oldE
            if dE < 0 or r < math.exp(-beta * dE):
                config[x, y, z, 0] = new0
                config[x, y, z, 1] = new1
                config[x, y, z, 2] = new2
    return config

# Wolf Cluster Algorithms 2D

@njit
def wolff_update_2d_ising(config, beta, N):
    
    # create array to store sites to be flipped (cluster membership)
    cluster = np.zeros((N, N), dtype=np.int8)
    
    # create stack to store site coordinates to be searched
    stack = np.empty((N*N, 2), dtype=np.int64)
    # pointer for number of sites left to search
    stack_ptr = 0
    
    # pick a random site 
    i = np.random.randint(0, N)
    j = np.random.randint(0, N)
    s0 = config[i, j]
    

    cluster[i, j] = 1
    stack[0, 0] = i; stack[0, 1] = j
    stack_ptr = 1
    
    p_add = 1.0 - math.exp(-2.0 * beta)

    while stack_ptr > 0:
        stack_ptr -= 1
        # retrieve next site coordinates
        i = stack[stack_ptr, 0]
        j = stack[stack_ptr, 1]
        # For each neighbor:
        for di, dj in ((1,0), (-1,0), (0,1), (0,-1)):
            ni = (i + di) % N
            nj = (j + dj) % N
            # if neighbor is not yet in the cluster and has the same spin as the seed:
            if cluster[ni, nj] == 0 and config[ni, nj] == s0:
                # add neighbor with probability p_add
                if np.random.random() < p_add:
                    config[ni, nj] = -config[ni, nj]
                
                    cluster[ni, nj] = 1
                    stack[stack_ptr, 0] = ni
                    stack[stack_ptr, 1] = nj
                    stack_ptr += 1
                    
                        
    return config


@njit
def wolff_update_2d_xy(config, beta, N):
    
    # Choose a random reflection direction (unit vector in 2D)
    phi_ref = np.random.uniform(0, 2*math.pi)
    
    # store components
    r0 = math.cos(phi_ref)
    r1 = math.sin(phi_ref)
    
    # create array to store what needs to be reflected
    cluster = np.zeros((N, N), dtype=np.int8)
    
    # create stack to store what site coordinates needs to be searches
    stack = np.empty((N*N, 2), dtype=np.int64)
    # create pointer for number of sites left to search
    stack_ptr = 0
    
    # pick random site
    i = np.random.randint(0, N)
    j = np.random.randint(0, N)
    
    # flip seed
    theta = config[i, j]
    config[i, j] = (2*phi_ref - theta) % (2*math.pi)
    
    cluster[i, j] = 1
    stack[0, 0] = i; stack[0, 1] = j
    stack_ptr = 1
    
    while stack_ptr > 0:
        stack_ptr -= 1
        # retrieve next site coords
        i = stack[stack_ptr, 0]
        j = stack[stack_ptr, 1]
        # For each neighbor:
        for di, dj in ((1,0), (-1,0), (0,1), (0,-1)):
            ni = (i+di) % N
            nj = (j+dj) % N
            if cluster[ni, nj] == 0:
                theta_i = config[i, j]
                theta_n = config[ni, nj]
                a_i = (math.cos(theta_i)*r0 + math.sin(theta_i)*r1)
                a_n = (math.cos(theta_n)*r0 + math.sin(theta_n)*r1)
                
                p_add = 1.0 - math.exp(min(0,2.0 * beta * a_i * a_n))
                
                if np.random.random() < p_add:
                    config[ni, nj] = (2*phi_ref - theta_n) % (2*math.pi)
                    cluster[ni, nj] = 1
                    stack[stack_ptr, 0] = ni
                    stack[stack_ptr, 1] = nj
                    stack_ptr += 1
                    
    return config

    
@njit
def wolff_update_2d_heisenberg(config, beta, N):
    
    # Choose a random reflection vector in R3 using spherical coords
    theta_r = np.random.uniform(0, 2*math.pi)  
    phi_r = np.random.uniform(0, math.pi)        
    
    # store components
    r0 = math.sin(phi_r) * math.cos(theta_r)
    r1 = math.sin(phi_r) * math.sin(theta_r)
    r2 = math.cos(phi_r)
    
    # create array to store sites 
    cluster = np.zeros((N, N), dtype=np.int8)
    
    # create stack to store site coordinates to be searched
    stack = np.empty((N*N, 2), dtype=np.int64)
    # pointer for number of sites left to search
    stack_ptr = 0
    
    # pick a random site as the seed
    i = np.random.randint(0, N)
    j = np.random.randint(0, N)
    
    # compute projection a = S · r for the seed spin
    a = config[i, j, 0]*r0 + config[i, j, 1]*r1 + config[i, j, 2]*r2
    
    #flip seed
    s0 = config[i, j, 0]
    s1 = config[i, j, 1]
    s2 = config[i, j, 2]
    config[i, j, 0] = s0 - 2 * a * r0
    config[i, j, 1] = s1 - 2 * a * r1
    config[i, j, 2] = s2 - 2 * a * r2
        
    # mark the seed as in the cluster and add it to the stack
    cluster[i, j] = 1
    stack[0, 0] = i; stack[0, 1] = j
    stack_ptr = 1
    
    
    while stack_ptr > 0:
        stack_ptr -= 1
        # retrieve next site coordinates
        i = stack[stack_ptr, 0]
        j = stack[stack_ptr, 1]
        # For each neighbor:
        for di, dj in ((1,0), (-1,0), (0,1), (0,-1)):
            ni = (i + di) % N
            nj = (j + dj) % N
            if cluster[ni, nj] == 0:

                a_i = config[i, j, 0]*r0 + config[i, j, 1]*r1 + config[i, j, 2]*r2
                a_n = config[ni, nj, 0]*r0 + config[ni, nj, 1]*r1 + config[ni, nj, 2]*r2
                
                p_add = 1.0 - math.exp(2.0 * beta * min(0,a_i * a_n))

                if np.random.random() < p_add:
                    
                    s0_i = config[ni, nj, 0]
                    s1_i = config[ni, nj, 1]
                    s2_i = config[ni, nj, 2]
                    a_val = s0*r0 + s1*r1 + s2*r2
                    config[ni, nj, 0] = s0_i - 2 * a_val * r0
                    config[ni, nj, 1] = s1_i - 2 * a_val * r1
                    config[ni, nj, 2] = s2_i - 2 * a_val * r2
                    
                    cluster[ni, nj] = 1
                    stack[stack_ptr, 0] = ni
                    stack[stack_ptr, 1] = nj
                    stack_ptr += 1
                        
    return config

#Wrapper for cluster update
@njit
def wolff_update_2d(config, beta, N, model):
    if model == 0:
        return wolff_update_2d_ising(config, beta, N)
    elif model == 1:
        return wolff_update_2d_xy(config, beta, N)
    elif model == 2:
        return wolff_update_2d_heisenberg(config, beta, N)
    

# 3D Cluster Updates
@njit
def wolff_update_3d_ising(config, beta, N):
    # create array to store sites to be flipped (cluster membership)
    cluster = np.zeros((N, N, N), dtype=np.int8)
    
    # create stack to store site coordinates to be searched
    stack = np.empty((N*N*N, 3), dtype=np.int64)
    stack_ptr = 0
    
    # pick a random site
    x = np.random.randint(0, N)
    y = np.random.randint(0, N)
    z = np.random.randint(0, N)
    s0 = config[x, y, z]
    
    # mark seed as in the cluster and add to stack
    cluster[x, y, z] = 1
    stack[0, 0] = x; stack[0, 1] = y; stack[0, 2] = z
    stack_ptr = 1
    
    p_add = 1.0 - math.exp(-2.0 * beta)
    
    # Grow the cluster
    while stack_ptr > 0:
        stack_ptr -= 1
        x = stack[stack_ptr, 0]
        y = stack[stack_ptr, 1]
        z = stack[stack_ptr, 2]
        # iterate over the six nearest neighbors in 3D
        for dx, dy, dz in ((1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)):
            nx = (x + dx) % N
            ny = (y + dy) % N
            nz = (z + dz) % N
            if cluster[nx, ny, nz] == 0 and config[nx, ny, nz] == s0:
                if np.random.random() < p_add:
                    # flip neighbor spin on the fly
                    config[nx, ny, nz] = -config[nx, ny, nz]
                    cluster[nx, ny, nz] = 1
                    stack[stack_ptr, 0] = nx
                    stack[stack_ptr, 1] = ny
                    stack[stack_ptr, 2] = nz
                    stack_ptr += 1
                    
    return config

@njit
def wolff_update_3d_xy(config, beta, N):
    # Choose a random reflection direction (unit vector in 2D, same for each site)
    phi_ref = np.random.uniform(0, 2*math.pi)
    r0 = math.cos(phi_ref)
    r1 = math.sin(phi_ref)
    
    # create array to store what needs to be reflected (cluster membership)
    cluster = np.zeros((N, N, N), dtype=np.int8)
    
    # create stack to store site coordinates to be searched
    stack = np.empty((N*N*N, 3), dtype=np.int64)
    stack_ptr = 0
    
    # pick a random site (seed)
    x = np.random.randint(0, N)
    y = np.random.randint(0, N)
    z = np.random.randint(0, N)
    theta = config[x, y, z]
    
    # flip seed unconditionally
    config[x, y, z] = (2*phi_ref - theta) % (2*math.pi)
    cluster[x, y, z] = 1
    stack[0, 0] = x; stack[0, 1] = y; stack[0, 2] = z
    stack_ptr = 1
    
    # Grow the cluster
    while stack_ptr > 0:
        stack_ptr -= 1
        x = stack[stack_ptr, 0]
        y = stack[stack_ptr, 1]
        z = stack[stack_ptr, 2]
        # iterate over the six nearest neighbors in 3D
        for dx, dy, dz in ((1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)):
            nx = (x + dx) % N
            ny = (y + dy) % N
            nz = (z + dz) % N
            
            if cluster[nx, ny, nz] == 0:
                theta_i = config[x, y, z]
                theta_n = config[nx, ny, nz]
                a_i = (math.cos(theta_i)*r0 + math.sin(theta_i)*r1)
                a_n = (math.cos(theta_n)*r0 + math.sin(theta_n)*r1)
                
                p_add = 1.0 - math.exp(min(0,2.0 * beta * a_i * a_n))
                
                if np.random.random() < p_add:
                    config[nx, ny, nz] = (2*phi_ref - theta_n) % (2*math.pi)
                    cluster[nx, ny, nz] = 1
                    stack[stack_ptr, 0] = nx
                    stack[stack_ptr, 1] = ny
                    stack[stack_ptr, 2] = nz
                    stack_ptr += 1    
                
                        
    return config


@njit
def wolff_update_3d_heisenberg(config, beta, N):
    # Choose a random reflection vector in R^3 using spherical coordinates
    theta_r = np.random.uniform(0, 2*math.pi)
    phi_r = np.random.uniform(0, math.pi)
    r0 = math.sin(phi_r)*math.cos(theta_r)
    r1 = math.sin(phi_r)*math.sin(theta_r)
    r2 = math.cos(phi_r)
    
    # create array to store sites (cluster membership)
    cluster = np.zeros((N, N, N), dtype=np.int8)
    
    # create stack to store site coordinates to be searched
    stack = np.empty((N*N*N, 3), dtype=np.int64)
    stack_ptr = 0
    
    # pick a random site as the seed
    x = np.random.randint(0, N)
    y = np.random.randint(0, N)
    z = np.random.randint(0, N)
    
    # compute projection a = S · r for the seed spin
    a = config[x, y, z, 0]*r0 + config[x, y, z, 1]*r1 + config[x, y, z, 2]*r2
    # flip seed unconditionally using the reflection operation
    s0 = config[x, y, z, 0]
    s1 = config[x, y, z, 1]
    s2 = config[x, y, z, 2]
    config[x, y, z, 0] = s0 - 2 * a * r0
    config[x, y, z, 1] = s1 - 2 * a * r1
    config[x, y, z, 2] = s2 - 2 * a * r2
    
    cluster[x, y, z] = 1
    stack[0, 0] = x; stack[0, 1] = y; stack[0, 2] = z
    stack_ptr = 1
    
    # Grow the cluster
    while stack_ptr > 0:
        stack_ptr -= 1
        x = stack[stack_ptr, 0]
        y = stack[stack_ptr, 1]
        z = stack[stack_ptr, 2]
        # iterate over the six nearest neighbors in 3D
        for dx, dy, dz in ((1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)):
            nx = (x + dx) % N
            ny = (y + dy) % N
            nz = (z + dz) % N
            if cluster[nx, ny, nz] == 0:

                a_i = config[x, y, z, 0]*r0 + config[x, y, z, 1]*r1 + config[x, y, z, 2]*r2
                a_n = config[nx, ny, nz, 0]*r0 + config[nx, ny, nz, 0]*r1 + config[nx, ny, nz, 0]*r2
                
                p_add = 1.0 - math.exp(2.0 * beta * min(0,a_i * a_n))

                if np.random.random() < p_add:
                    
                    s0_i = config[nx, ny, nz, 0]
                    s1_i = config[nx, ny, nz, 1]
                    s2_i = config[nx, ny, nz, 2]
                    a_val = s0*r0 + s1*r1 + s2*r2
                    config[nx, ny, nz, 0] = s0_i - 2 * a_val * r0
                    config[nx, ny, nz, 1] = s1_i - 2 * a_val * r1
                    config[nx, ny, nz, 2] = s2_i - 2 * a_val * r2
                    
                    cluster[nx, ny, nz, 0] = 1
                    stack[stack_ptr, 0] = nx
                    stack[stack_ptr, 1] = ny
                    stack[stack_ptr, 2] = nz
                    stack_ptr += 1
                
    return config


@njit
def wolff_update_3d(config, beta, N, model):
    if model == 0:
        return wolff_update_3d_ising(config, beta, N)
    elif model == 1:
        return wolff_update_3d_xy(config, beta, N)
    elif model == 2:
        return wolff_update_3d_heisenberg(config, beta, N)

# ==========================================================
# COMBINED MEASUREMENTS (energy, magnetisation, correlation)
# ==========================================================
@njit
def measure2d_all(config, N, model):
    R = N // 2
    energy = 0.0
    if model == 0:
        mag = 0.0
        for i in range(N):
            for j in range(N):
                S = config[i, j]
                energy += -S * (config[i, (j+1)%N] + config[(i+1)%N, j])
                mag += S
        corr = np.zeros(R+1, dtype=np.float64)
        count = np.zeros(R+1, dtype=np.int64)
        for i in range(N):
            for j in range(N):
                for r in range(R+1):
                    corr[r] += config[i, j] * config[(i+r)%N, j]
                    count[r] += 1
        for r in range(R+1):
            if count[r] > 0:
                corr[r] /= count[r]
        return energy, mag, corr
    elif model == 1:
        sum_cos = 0.0
        sum_sin = 0.0
        for i in range(N):
            for j in range(N):
                theta = config[i, j]
                energy += - (math.cos(theta - config[i, (j+1)%N]) +
                            math.cos(theta - config[(i+1)%N, j]))
                sum_cos += math.cos(theta)
                sum_sin += math.sin(theta)
        mag = math.sqrt(sum_cos*sum_cos + sum_sin*sum_sin)
        corr = np.zeros(R+1, dtype=np.float64)
        count = np.zeros(R+1, dtype=np.int64)
        for i in range(N):
            for j in range(N):
                for r in range(R+1):
                    corr[r] += math.cos(config[i, j] - config[(i+r)%N, j])
                    count[r] += 1
        for r in range(R+1):
            if count[r] > 0:
                corr[r] /= count[r]
        return energy, mag, corr
    elif model == 2:
        sum0 = 0.0; sum1 = 0.0; sum2 = 0.0
        for i in range(N):
            for j in range(N):
                s0 = config[i, j, 0]
                s1 = config[i, j, 1]
                s2 = config[i, j, 2]
                energy += - ((s0 * config[i, (j+1)%N, 0] + s1 * config[i, (j+1)%N, 1] + s2 * config[i, (j+1)%N, 2]) +
                           (s0 * config[(i+1)%N, j, 0] + s1 * config[(i+1)%N, j, 1] + s2 * config[(i+1)%N, j, 2]))
                sum0 += s0; sum1 += s1; sum2 += s2
        mag = math.sqrt(sum0*sum0 + sum1*sum1 + sum2*sum2)
        corr = np.zeros(R+1, dtype=np.float64)
        count = np.zeros(R+1, dtype=np.int64)
        for i in range(N):
            for j in range(N):
                for r in range(R+1):
                    s0 = config[i, j, 0]
                    s1 = config[i, j, 1]
                    s2 = config[i, j, 2]
                    t0 = config[(i+r)%N, j, 0]
                    t1 = config[(i+r)%N, j, 1]
                    t2 = config[(i+r)%N, j, 2]
                    corr[r] += s0*t0 + s1*t1 + s2*t2
                    count[r] += 1
        for r in range(R+1):
            if count[r] > 0:
                corr[r] /= count[r]
        return energy, mag, corr

@njit
def measure3d_all(config, N, model):
    R = N // 2
    energy = 0.0
    if model == 0:
        mag = 0.0
        for x in range(N):
            for y in range(N):
                for z in range(N):
                    S = config[x, y, z]
                    energy += -S * (config[(x+1)%N, y, z] + config[x, (y+1)%N, z] + config[x, y, (z+1)%N])
                    mag += S
        corr = np.zeros(R+1, dtype=np.float64)
        count = np.zeros(R+1, dtype=np.int64)
        for x in range(N):
            for y in range(N):
                for z in range(N):
                    for r in range(R+1):
                        corr[r] += config[x, y, z] * config[(x+r)%N, y, z]
                        count[r] += 1
        for r in range(R+1):
            if count[r] > 0:
                corr[r] /= count[r]
        return energy, mag, corr
    elif model == 1:
        sum_cos = 0.0; sum_sin = 0.0
        for x in range(N):
            for y in range(N):
                for z in range(N):
                    theta = config[x, y, z]
                    energy += - (math.cos(theta - config[(x+1)%N, y, z]) +
                                math.cos(theta - config[x, (y+1)%N, z]) +
                                math.cos(theta - config[x, y, (z+1)%N]))
                    sum_cos += math.cos(theta)
                    sum_sin += math.sin(theta)
        mag = math.sqrt(sum_cos*sum_cos + sum_sin*sum_sin)
        corr = np.zeros(R+1, dtype=np.float64)
        count = np.zeros(R+1, dtype=np.int64)
        for x in range(N):
            for y in range(N):
                for z in range(N):
                    for r in range(R+1):
                        corr[r] += math.cos(config[x, y, z] - config[(x+r)%N, y, z])
                        count[r] += 1
        for r in range(R+1):
            if count[r] > 0:
                corr[r] /= count[r]
        return energy, mag, corr
    elif model == 2:
        sum0 = 0.0; sum1 = 0.0; sum2 = 0.0
        for x in range(N):
            for y in range(N):
                for z in range(N):
                    s0 = config[x, y, z, 0]
                    s1 = config[x, y, z, 1]
                    s2 = config[x, y, z, 2]
                    energy += - ((s0 * config[(x+1)%N, y, z, 0] + s1 * config[(x+1)%N, y, z, 1] + s2 * config[(x+1)%N, y, z, 2]) +
                                (s0 * config[x, (y+1)%N, z, 0] + s1 * config[x, (y+1)%N, z, 1] + s2 * config[x, (y+1)%N, z, 2]) +
                                (s0 * config[x, y, (z+1)%N, 0] + s1 * config[x, y, (z+1)%N, 1] + s2 * config[x, y, (z+1)%N, 2]))
                    sum0 += s0; sum1 += s1; sum2 += s2
        mag = math.sqrt(sum0*sum0 + sum1*sum1 + sum2*sum2)
        corr = np.zeros(R+1, dtype=np.float64)
        count = np.zeros(R+1, dtype=np.int64)
        for x in range(N):
            for y in range(N):
                for z in range(N):
                    for r in range(R+1):
                        s0 = config[x, y, z, 0]
                        s1 = config[x, y, z, 1]
                        s2 = config[x, y, z, 2]
                        t0 = config[(x+r)%N, y, z, 0]
                        t1 = config[(x+r)%N, y, z, 1]
                        t2 = config[(x+r)%N, y, z, 2]
                        corr[r] += s0*t0 + s1*t1 + s2*t2
                        count[r] += 1
        for r in range(R+1):
            if count[r] > 0:
                corr[r] /= count[r]
        return energy, mag, corr

# ==========================================================
# SIMULATION ROUTINE
# ==========================================================
def run_simulation(config, beta, eqSteps, mcSteps, N, dim, model, delta, update_type):
    """
    Runs equilibration then measurement sweeps.
    If update_type == 'local', local Metropolis moves are used.
    If update_type == 'cluster', Wolff cluster updates are used.
    Measurements (energy, magnetisation, correlation) are computed in one pass.
    """
    # Convert model to numeric code: 0=ising, 1=xy, 2=heisenberg.
    if model == 'ising':
        mcode = 0
    elif model == 'xy':
        mcode = 1
    elif model == 'heisenberg':
        mcode = 2
    else:
        raise ValueError("Unknown model")
    
    # Equilibration:
    for _ in range(eqSteps):
        if update_type == 'local':
            if dim == 2:
                mcmove2d(config, beta, N, mcode, delta)
            else:
                mcmove3d(config, beta, N, mcode, delta)
        elif update_type == 'cluster':
            if dim == 2:
                wolff_update_2d(config, beta, N, mcode)
            else:
                wolff_update_3d(config, beta, N, mcode)
    # Measurement:
    m_i = np.empty(mcSteps, dtype=np.float64)
    e_i = np.empty(mcSteps, dtype=np.float64)
    R = N // 2
    corr_sum = np.zeros(R+1, dtype=np.float64)
    for i in range(mcSteps):
        if update_type == 'local':
            if dim == 2:
                mcmove2d(config, beta, N, mcode, delta)
            else:
                mcmove3d(config, beta, N, mcode, delta)
        elif update_type == 'cluster':
            if dim == 2:
                wolff_update_2d(config, beta, N, mcode)
            else:
                wolff_update_3d(config, beta, N, mcode)
        if dim == 2:
            e, m, corr = measure2d_all(config, N, mcode)
        else:
            e, m, corr = measure3d_all(config, N, mcode)
        e_i[i] = e
        m_i[i] = m
        corr_sum += corr
    corr_avg = corr_sum / mcSteps
    return m_i, e_i, corr_avg

# ==========================================================
# SIMULATION WRAPPER (for multiprocessing)
# ==========================================================
def simulate_temp(args, output_folder, dim, model, delta, update_type):
    """
    Runs simulation for given N and T_value and saves data.
    """
    N, T_value, eqSteps, mcSteps = args
    beta = 1.0 / T_value
    config = initialstate(N, dim, model)
    m_i, e_i, corr_avg = run_simulation(config, beta, eqSteps, mcSteps, N, dim, model, delta, update_type)
    filename = os.path.join(output_folder, f"run-T{T_value:.3f}N{N}D{dim}-{model}-{update_type}.npz")
    np.savez(filename,
             energy=e_i,
             magnetisation=m_i,
             correlation=corr_avg)
    print(f"Finished: T={T_value:.3f}, N={N}, dim={dim}, model={model}, update={update_type}, avg m/site={m_i.mean()/(N**dim):.3f}")
    return T_value, N, m_i, e_i, corr_avg

# ==========================================================
# DATA CREATION LOOP
# ==========================================================
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

# ==========================================================
# MAIN EXECUTION
# ==========================================================
if __name__ == '__main__':
    output_folder = "data10"
    nt          = 50                      # Number of temperature points
    n_list      = [16, 32, 64]            # Lattice sizes
    eqSteps     = 1024 * 10               # Equilibration sweeps
    mcSteps     = 1024 * 10               # Measurement sweeps
    T_arr       = np.linspace(3.5, 5.5, nt) # Temperature array
    dim         = 3                       # 2 or 3
    model       = 'xy'                    # 'ising', 'xy', or 'heisenberg'
    delta       = 0.3                     # For local moves (not used in cluster)
    update_type = 'cluster'               # Choose 'local' or 'cluster'
    
    create_data(output_folder, nt, n_list, eqSteps, mcSteps, T_arr, dim, model, delta, update_type)
