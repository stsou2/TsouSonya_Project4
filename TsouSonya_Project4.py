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
        '''
        Creates a tri-diagonal matrix.

        Args:
        
        N (int): number of rows/cols for square matrix
        b (float): value of one below the diagonal
        d (float): value of the diagonal
        a (float): value of one above the diagonal
        
        Returns:
        
        A (array): NxN matrix with d on the diagonal, b one below 
        the diagonal, and a, one above'''

        A = np.eye(N,k=1)*a + np.eye(N)*d + np.eye(N,k=-1)*b
        return A

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
    
    ### parameters
    L = length # The system extends from x=-L/2 to x=L/2
    V = potential
    h = L/(nspace) # Grid spacing for periodic boundary conditions

    # constants
    hbar = 1
    m = 0.5

    ### Create H matrix with periodic boundary conditions.

    # Tridiagonal matrix
    # For the given values of hbar and m, the coeff works out to just -1/h**2
    # but I am putting the full thing in for clarity
    H = (-hbar**2/(2*m*h**2))*make_tridiagonal(nspace, 1, -2, 1)  

    # Adding potential
    if not V: # if potential is empty
        pass
    else: # if potential has values in the array
        for v in V:
            H[v, v] = 1* ((h**2)*2*m)/(-hbar**2) # accounting for coefficient; index known because on main diagonal
    H[0, -1] = 1   # Top right corner = b (in tridiagonal) for BC
    H[-1, 0] = 1  # Bottom left corner = a (in tridiagonal) for BC
    
    # methods  -referring to Eqns 9.32 and 9.40 in the textbook
    if method == 'ftcs': # FTCS method
        A = np.eye(nspace) - (1j*tau/hbar)*H
    
    elif  method == 'crank': # Crank-Nicolson method
        A = np.linalg.inv(np.eye(nspace) + (1j*tau/hbar)*H)*(np.eye(nspace) - (1j*tau/hbar)*H)
      
    else:
        raise ValueError("Invalid method. Please enter either 'ftcs' or 'crank' as the method param.")
    
    ### Stability
    eigenval = spectral_radius(A)
    if eigenval-1 > 1e-10:
        print("Solution will be unstable.")
    else:
        print("Solution will be stable.")
    
    # creating spatial and time grids
    x_grid = np.linspace(-L/2, L/2, nspace, endpoint=False)  # spatial grid from -L/2 to L/2
    t_grid = np.arange(0,ntime) * tau  # time grid 
    
    # Initializing psi
    psi = np.zeros((nspace, ntime))
    psi[:, 0] = make_gaussIC(x_i=x_grid)  # IC

    for n in range(1, ntime):
        psi[:,n] = np.dot(A, psi[:,n-1])

    # Finding prob, ie psi times psi_conjugate
    prob = psi * np.conjugate(psi)

    return psi, x_grid, t_grid, prob

def sch_plot(sch_arrays, type = 'psi', save = False):
    '''
    Plots output of sch_eqn().

    Args:
    outputs (1x4 tuple): tuple of sch_eqn outputs (psi, x_grid, t_grid, prob) 
    type (str): type of plot, either 'psi' (plot of the real part of ψ(x) at time t)
        or 'prob' (plot of the particle probability density ψ ψ*(x) at a specific time). Defaults to psi
    save (bool): Option to save to file. Defaults to False
    '''
    
    return

