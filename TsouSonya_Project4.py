import numpy as np
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
        gaussIC = (1/(np.sqrt(sigma_0*np.sqrt(np.pi))))*(np.exp(1j*k_0*x_i))*np.exp((-(x_i-x_0)**2)/(2*sigma_0**2))
        return gaussIC
    
    # From Lab 11
    
    ### parameters
    # variable
    L = length # The system extends from x=-L/2 to x=L/2
    V = potential
    #nspace = nspace - 1 # Technically there are nspace - 1 points to calculate;
    # >> in light of not changing every single proceeding nspace variable I just did this to fix it
    h = L/(nspace) # Grid spacing for periodic boundary conditions
    sigma0 = wparam[0]
    x0 = wparam[1]
    k0 = wparam[2]

    # constants
    hbar = 1
    m = 0.5

    ### Create H matrix with periodic boundary conditions.

    # Tridiagonal matrix
    H = make_tridiagonal(nspace, 1, -2, 1)  

    # Adding potential
    if not V: # if potential is empty
        pass
    else: # if potential has values in the array
        for v in V:
            H[v, v] = 1* ((h**2)*2*m)/(-(hbar**2)) # accounting for coefficient; index known because on main diagonal
    H[0, -1] = 1   # Top right corner = b (in tridiagonal) for BC
    H[-1, 0] = 1  # Bottom left corner = a (in tridiagonal) for BC

    # fixing coefficient
    # For the given values of hbar and m, the coeff works out to just -1/h**2
    # but I am putting the full thing in for clarity. Factored elsewhere as appropriate
    coeff = ((-hbar**2)/(2*m*h**2))

    # methods  -referring to Eqns 9.32 and 9.40 in the textbook
    if method == 'ftcs': # FTCS method
        A = np.eye(nspace) - (1j*tau/hbar)*coeff*H
    
    elif  method == 'crank': # Crank-Nicolson method
        A = np.matmul(np.linalg.inv(np.eye(nspace) + ((1j*tau)/(2*hbar))*coeff*H), (np.eye(nspace) - ((1j*tau)/(2*hbar))*coeff*H))
    
    else:
        raise ValueError("Invalid method. Please enter either 'ftcs' or 'crank' as the method param.")
    
    
    #print(A)
    ### Stability
    eigenval = spectral_radius(A)
    # Must find abs value.
    if abs(eigenval)-1 > 1e-10:
        print("Solution will be unstable.")
    else:
        print("Solution will be stable.")


    ### Solving
         
    # creating spatial and time grids
    x_grid = np.linspace(-L/2, L/2, nspace, endpoint=True)  # spatial grid from -L/2 to L/2
    t_grid = np.arange(0, ntime) * tau  # time grid 
    
    # Initializing psi
    psi = np.zeros((nspace, ntime), dtype=complex)
    psi[:, 0] = make_gaussIC(sigma0, k0, x0, x_i=x_grid)  # IC

    for n in range(1, ntime):
        psi[:,n] = np.matmul(A, psi[:,n-1])

    # Finding total prob, ie psi times psi_conjugate
    prob = np.zeros(ntime, dtype = complex)
    prob = np.sum(psi * np.conjugate(psi), axis=0)
    
    return psi, x_grid, t_grid, prob



def sch_plot(sch_arrays, time = 0, type = 'psi', plotshow = True, save = False):
    '''
    Plots output of sch_eqn().

    Args:
    outputs (1x4 tuple): tuple of sch_eqn outputs (psi, x_grid, t_grid, prob) 
    time (float-like): time point in grid at which plot is desired. May be approximated.
    type (str): type of plot, either 'psi' (plot of the real part of ψ(x) at time t)
        or 'prob' (plot of the particle probability density ψ ψ*(x) at a specific time). Defaults to psi
    plotshow (bool): Option to display the plot. Defaults to True
    save (bool): Option to save to file. Defaults to False
    '''
    psi = np.real(sch_arrays[0]) #taking real only
    x_grid = sch_arrays[1]
    t_grid = sch_arrays[2]
    prob = sch_arrays[0]*np.conjugate(sch_arrays[0])

    ### Match given time ooint to closest match in time grid, rounding down 
    t_index = np.searchsorted(t_grid, time, side = 'left') # left side means t_grid[i-1] < time <= t_grid[i], ie rounding down

    ### Plotting
    fig, ax = plt.subplots()
    t_label = np.round(t_grid[t_index], 2)
    # psi plot
    if type == 'psi':
        ax.plot(x_grid, psi[:, t_index])
        ax.set(xlabel = 'x', ylabel = 'psi(x)', title = f'Plot of Schroedinger Wavefunction psi(x) at Time t={t_label}')
    # prob plot
    elif type == 'prob':
        ax.plot(x_grid, prob[:, t_index])
        ax.set(xlabel = 'x', ylabel = f'P(x, t={t_label})', title = f'Probability Density at Time t={t_label}')
    else:
        raise ValueError("Please choose either 'psi' or 'prob' as type.")
    
    plt.grid(True)

    if plotshow == True:
        plt.show()
    else:
        pass

    if save == True:
        plt.savefig(f"sch_plot_{type}.png")
    else:
        pass

    return

#print(sch_eqn(nspace = 30, ntime = 500, tau = 1.0, method = 'ftcs', length = 100)[3])
sch_plot(sch_arrays=sch_eqn(nspace = 30, ntime = 500, tau = 1.0, method = 'ftcs', length = 100), time = 10, type = 'psi')


# fig, ax = plt.subplots()

# #for i in np.linspace(0, 500, 6, endpoint = True):
# for i in [0, 10]:
#     psi, x_grid, t_grid, prob =sch_eqn(nspace = 100, ntime = 500, tau = 1.0, method = 'crank', length = 100)
#     psi = np.real(psi) #taking real only
#     ### Plotting
#     ax.plot(x_grid, psi[:, i], label = 'i')

# plt.legend()
# plt.show()


