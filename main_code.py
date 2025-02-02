# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from __future__ import division
import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt
from numba import njit
import scipy.stats 
import tqdm

#----------------------------------------------------------------------
##  BLOCK OF FUNCTIONS USED IN THE MAIN CODE
#----------------------------------------------------------------------


def initialstate(N):   
    ''' generates a random spin configuration for initial condition'''
    state = 2*np.random.randint(2, size=(N,N))-1
    return state

@njit
def mcmove(config, beta, N):
    '''Monte Carlo move using Metropolis algorithm '''
    for i in range(N):
        for j in range(N):
                a = np.random.randint(0, N)
                b = np.random.randint(0, N)
                s =  config[a, b]
                nb = config[(a+1)%N,b] + config[a,(b+1)%N] + config[(a-1)%N,b] + config[a,(b-1)%N]
                cost = 2*s*nb
                if cost < 0:
                    s *= -1
                elif rand() < np.exp(-cost*beta):
                    s *= -1
                config[a, b] = s
    return config

@njit
def calcEnergy(config, N):
    '''Energy of a given configuration'''
    energy = 0
    for i in range(len(config)):
        for j in range(len(config)):
            S = config[i,j]
            nb = config[(i+1)%N, j] + config[i,(j+1)%N] + config[(i-1)%N, j] + config[i,(j-1)%N]
            energy += -nb*S
    return energy/4.

@njit
def calcMag(config):
    '''Magnetization of a given configuration'''
    mag = np.sum(config)
    return mag


## change these parameters for a smaller (faster) simulation 
nt      = 10      #  number of temperature points
n       = [8,16,32]         #  size of the lattice, N x N
eqSteps = 1024       #  number of MC sweeps for equilibration
mcSteps = 1024*10       #  number of MC sweeps for calculation

T       = np.linspace(1.53, 3.28, nt)  #good spread tro show near temperature 3.28
E,M,C,X = np.zeros(nt), np.zeros(nt), np.zeros(nt), np.zeros(nt)
#n1, n2  = 1.0/(mcSteps*n*N), 1.0/(mcSteps*mcSteps*N*N) 
# divide by number of samples, and by system size to get intensive values



#----------------------------------------------------------------------
#  MAIN PART OF THE CODE
#----------------------------------------------------------------------



def create_data():
    M_final = []
    E_final = []
    
    for N in n:
        M = []
        E = []
        
        for tt in range(nt):
            print(T[tt],N)
            E1 = M1 = E2 = M2 = 0
            m_i = np.zeros(mcSteps)
            e_i = np.zeros(mcSteps)
            config = initialstate(N)
            iT=1.0/T[tt]; iT2=iT*iT;
            
            
            for i in range(eqSteps):         # equilibrate
                mcmove(config, iT, N)           # Monte Carlo moves
        
            # for i in tqdm.tqdm(range(mcSteps)):
            for i in range(mcSteps):
                mcmove(config, iT, N)           
                Ene = calcEnergy(config, N)     # calculate the energy
                Mag = calcMag(config)        # calculate the magnetisation
                m_i[i]= Mag
                e_i[i] = Ene
                
                
            print(T[tt],N,m_i.mean()/N**2)
            # M.append(m_i)
            # E.append(e_i)
            np.savez(f"data/run-T{T[tt]}N{N}.npz",energy=e_i,magnetisation=m_i)
            #E[tt] = n1*E1
            #M[tt] = n1*M1
            #C[tt] = (n1*E2 - n2*E1*E1)*iT2
            #X[tt] = (n1*M2 - n2*M1*M1)*iT
        # M_final.append(M)
        # E_final.append(E)      
    
    # return M_final, E_final

    
create_data()
# M_final, E_final = creat_data()


def stat_plot_sizes(stat_func, stat_name="Statistic"):
    """
    stat_func: a function like kurtosis, np.mean, etc.
    stat_name: label to show on the y-axis
    """
    
    # --- Plot Energy statistic vs Temperature ---
    
    plt.figure(figsize=(8, 5))
    for a in range(3):
        # Compute the statistic for each temperature
        stat_values_E = [stat_func(energy_samples) for energy_samples in E_final[a]]
        
        # Plot
        plt.plot(T, stat_values_E, marker='o', label=f'N={n[a]}')
    
    plt.xlabel("Temperature (T)", fontsize=14)
    plt.ylabel(f"{stat_name} of Energy", fontsize=14)
    plt.title(f"{stat_name} of Energy vs Temperature", fontsize=16)
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # --- Plot Magnetization statistic vs Temperature ---
    plt.figure(figsize=(8, 5))
    for a in range(3):
        # Compute the statistic for each temperature
        stat_values_M = [stat_func(mag_samples) for mag_samples in M_final[a]]
        
        # Plot
        plt.plot(T, stat_values_M, marker='s', label=f'N={n[a]}')
    
    plt.xlabel("Temperature (T)", fontsize=14)
    plt.ylabel(f"{stat_name} of Magnetization", fontsize=14)
    plt.title(f"{stat_name} of Magnetization vs Temperature", fontsize=16)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_iterations_for_temperature(Final_list, T, temp_index, size_index, property_name):
    """
    Plots either Magnetization or Energy vs iteration for all lattice sizes
    at a chosen temperature index.
    
    Parameters:
    -----------
    M_final      : list of lists of lists
                   M_final[i][j] is the magnetization time-series for 
                   lattice size n_list[i] at T[j].
    E_final      : list of lists of lists
                   E_final[i][j] is the energy time-series for 
                   lattice size n_list[i] at T[j].
    n_list       : list of lattice sizes used (e.g. [8, 16, 32])
    T            : array or list of temperatures
    temp_index   : int
                   Index in T for which we want to plot property vs iteration
    property_name: str, optional
                   'Magnetization' or 'Energy' (default is 'Magnetization')
    """
    # Create figure
    plt.figure(figsize=(8, 5))
    
    
    data = M_final[size_index][temp_index]
    ylabel = property_name

    # Plot data vs iteration
    plt.plot(range(len(data)), data, label=f'N = {n[size_index]}')
    
    # Labeling
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(f"{ylabel} vs. Iteration at T = {T[temp_index]:.2f}", fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.show()







# plot_iterations_for_temperature(M_final, T, 2, 2, "Magnetisation")
#stat_plot_sizes(kurtosis, "Kurtosis")


















# class Ising():
#     ''' Simulating the Ising model '''    
#     ## monte carlo moves
#     def mcmove(self, config, N, beta):
#         ''' This is to execute the monte carlo moves using 
#         Metropolis algorithm such that detailed
#         balance condition is satisified'''
#         for i in range(N):
#             for j in range(N):            
#                     a = np.random.randint(0, N)
#                     b = np.random.randint(0, N)
#                     s =  config[a, b]
#                     nb = config[(a+1)%N,b] + config[a,(b+1)%N] + config[(a-1)%N,b] + config[a,(b-1)%N]
#                     cost = 2*s*nb
#                     if cost < 0:	
#                         s *= -1
#                     elif rand() < np.exp(-cost*beta):
#                         s *= -1
#                     config[a, b] = s
#         return config
    
#     def simulate(self):   
#         ''' This module simulates the Ising model'''
#         N, temp     = 64, .4        # Initialse the lattice
#         config = 2*np.random.randint(2, size=(N,N))-1
#         f = plt.figure(figsize=(15, 15), dpi=80);    
#         self.configPlot(f, config, 0, N, 1);
        
#         msrmnt = 1001
#         for i in range(msrmnt):
#             self.mcmove(config, N, 1.0/temp)
#             if i == 1:       self.configPlot(f, config, i, N, 2);
#             if i == 4:       self.configPlot(f, config, i, N, 3);
#             if i == 32:      self.configPlot(f, config, i, N, 4);
#             if i == 100:     self.configPlot(f, config, i, N, 5);
#             if i == 1000:    self.configPlot(f, config, i, N, 6);
                 
                    
#     def configPlot(self, f, config, i, N, n_):
#         ''' This modules plts the configuration once passed to it along with time etc '''
#         X, Y = np.meshgrid(range(N), range(N))
#         sp =  f.add_subplot(3, 3, n_ )  
#         plt.setp(sp.get_yticklabels(), visible=False)
#         plt.setp(sp.get_xticklabels(), visible=False)      
#         plt.pcolormesh(X, Y, config, cmap=plt.cm.RdBu);
#         plt.title('Time=%d'%i); plt.axis('tight')    
#     plt.show()

# rm = Ising()

# rm.simulate()