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
    # From Lab 10:
    def make_tridiagonal(N, b, d, a):
        """
        Creates a tridiagonal matrix of size N x N with with d on the diagonal, b one below
        the diagonal, and a, one above.
        
        Parameters:
            N: The size of the matrix (N x N).
            b: The value for the diagonal below the main diagonal.
            d: The value for the main diagonal.
            a : The value for the diagonal above the main diagonal.
        
        Returns:
            A tridiagonal matrix with the given parameters.
        """
        
        #main diagonal
        A = d * np.eye(N)
        #below diagonal
        A += b * np.eye(N, k=-1)
        #above diagonal
        A += a * np.eye(N, k=1)
        
        return A

    def spectral_radius(A):
        """Calculates the maxium absolute eigenvalue of a 2-D array A

        Args:
            A : 2D array from part 1

        Returns:
            maximum absolute value of eigenvalues
        """    
        
        #determine the eigenvalues, only get that value from the tuple
        eigenvalues, _ = np.linalg.eig(A)
        
        #determine the maxiumum absolute value
        max_value = max(abs(eigenvalues))
        
        return max_value
    
    def make_gaussIC(sigma_0, k_0, x_0, x_i):
        """
        Function initialize Gaussian wave packet function for free particle IC.

        Args:
        sigma_0 (float): width of the wave packet
        k_0 (float): average wave number
        x_0 (float): x coordinate about which particle is localized 
        x_i (array-like): spatial grid points over which to apply IC

        Returns: 
        gaussIC: Gaussian wave packet function for psi(x, t=0)

        """
        gaussIC = (1/(np.sqrt(sigma_0*np.sqrt(np.pi))))*(np.exp(1j*k_0*x_i))*(np.exp(-(x_i-x_0)**2)/(2*sigma_0**2))
        return gaussIC
    
    # From Lab 11
    
    # Extract parameters
    L = length # The system extends from x=-L/2 to x=L/2
    V = potential
    h = L/(nspace)       # Grid spacing for periodic boundary conditions

    #tau_crit = h / c # critical time step
    #tau = tau_rel * tau_crit


    # Create H matrix with periodic boundary conditions.

    B = make_tridiagonal(nspace, -1, 0, 1)  # Tridiagonal matrix
    B[0, -1] = -1   # Top right for BC
    B[-1, 0] = 1  # Bottom left for BC


    #method.
    if method == 1 :      ### FTCS method ###
        # A is needed to understand how a evolves over time
        A = np.eye(nspace) - (c * tau / (2 * h)) * B #also reffering to p.220

    elif  method == 2 :   ### Lax method ###
        # need C: as in the fomula there is 1/2 C let's set C with this 1/2 already inside (with 0.5 instead of 1)
        C = make_tridiagonal(nspace, 0.5, 0, 0.5)

        # BC
        C[0, -1] = 0.5  # First row, last element
        C[-1, 0] = 0.5  # Last row, first element
        A = C - (c * tau / (2 * h)) * B

    else:
        raise ValueError("Invalid method. Please type 1 for 'FTCS' or  2 for'Lax'.")
    
    # Sonya
    # Stability
    eigenval = spectral_radius(A)
    if eigenval-1 > 1e-10:
        print("Solution will be unstable.")
    else:
        print("Solution will be stable.")
    
    # returns are a x and t

    # creating spatial and time grids
    x_grid = np.linspace(-L/2, L/2, nspace, endpoint=False)  # spatial grid from -L/2 to L/2
    t_grid = np.arange(0,ntime) * tau  # time grid 
    
    # Initializing psi
    psi = np.zeros((nspace, ntime))
    psi[:, 0] = make_gaussIC(x_i=x_grid)  # InC

    for n in range(1, ntime):
        psi[:,n] = np.dot(A, psi[:,n-1])

    # Finding prob, ie psi times psi_conjugate
    prob = psi * np.conjugate(psi)

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