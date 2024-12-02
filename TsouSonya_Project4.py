import numpy as np
import scipy as sc
import matplotlib.pyplot as plt

def sch_eqn(nspace, ntime, tau, method='ftcs', length=200, potential = [], wparam = [10, 0, 0.5]):
    '''
    Solves the 1D time-dependant Schroedinger equation.

    Args:
    nspace (int): number of spatial grid points
    ntime (int): number of time steps to evaluate
    tau (float): time step
    method (str): either 'ftcs' or 'crank'; numerical method to iterate with 
    length (float): size of spatial grid. Defaults to 200 (grid extends from -100 to +100)
    potential (1D array): the spatial index values at which the potential V(x) should be set to 1. Default to empty. 
    wparam (1x3 array-like): list of parameters for initial condition [sigma0, x0, k0]. Default [10, 0, 0.5].
    
    Returns:
    psi (2D array): ψ(x,t) values
    x_grid (1D array): x (spatial) grid values
    t_grid (1D array): time grid values
    prob (1D array): total probability computed for each timestep (conserved)

    '''
    # From Lab 10, to check FTCS stability
    def spectral_radius(A):
        '''
        Computes the eigenvalues of an input 2-D array A and returns the eigenvalue with greatest magnitude
        
        Args:
        A (array): 2D square input array

        Returns:
        lambda_max (float): eigenvalue with greatest magnitude
        '''

        (l,m) = np.linalg.eig(A)
        lambda_max = l[np.argmax(abs(l))]

        return lambda_max
    
    return psi, x_grid, t_grid, prob

def sch_plot(sch_arrays, type = 'psi', save = False):
    '''
    Plots output of sch_eqn().

    Args:
    sch_arrays (1x3 array): array of sch_eqn outputs; 
        either [psi, x_grid, t_grid] OR [prob, x_grid, t_grid] depending on desired plot type
    type (str): type of plot, either 'psi' (plot of the real part of ψ(x) at time t)
        or 'prob' (plot of the particle probability density ψ ψ*(x) at a specific time). Defaults to psi
    save (bool): Option to save to file. Defaults to False
    '''

    return