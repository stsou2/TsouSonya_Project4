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
    # From Lab 10/11:
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
    

    # Taken from Lab 11, and textbook: Numerical Methods for Physics, Second Edition, Revised (Python) by Alejandro L. Garcia (2017)

    ### Parameters
    # variable
    L = length # The system extends from x=-L/2 to x=L/2
    V = potential
    h = L/(nspace) # Grid spacing for periodic boundary conditions. Textbook appears to use nspace division instead of nspace-1
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
            H[v, v] = -2 + 1* ((h**2)*2*m)/(-(hbar**2)) # accounting for coefficient, index known because on main diagonal
    H[0, -1] = 1   # Top right corner = b (in tridiagonal) for BC
    H[-1, 0] = 1  # Bottom left corner = a (in tridiagonal) for BC

    # Fixing coefficient
        # For the given values of hbar and m, the coeff works out to just -1/h**2
        # but I am putting the full thing in for clarity. Factored elsewhere as appropriate
    coeff = ((-hbar**2)/(2*m*h**2))


    ### Methods, building A  -referring to Eqns 9.32 and 9.40 in the textbook
    if method == 'ftcs': # FTCS method
        A = np.eye(nspace) - (1j*tau/hbar)*coeff*H

        ## Matrix stability check
        eigenval = spectral_radius(A) # eigenvalue of largest magnitude
        # Converting back to magnitude for stability check.
        if abs(eigenval)-1 > 1e-10: # Unstable for magnitude > 1. Accounts for numerical error.
            raise ValueError("FTCS solution will be unstable. Please choose a smaller timestep (tau) or the 'crank' method.")
        else:
            print("FTCS solution will be stable.")
    
    elif  method == 'crank': # Crank-Nicolson method. Unconditionally stable.
        A = np.matmul(np.linalg.inv(np.eye(nspace) + ((1j*tau)/(2*hbar))*coeff*H), (np.eye(nspace) - ((1j*tau)/(2*hbar))*coeff*H))
    
    else:
        raise ValueError("Invalid method. Please enter either 'ftcs' or 'crank' as the method param.")
    

    ### Solving
    # Creating spatial and time grids
    x_grid = np.linspace(-L/2, L/2, nspace, endpoint=True)  # spatial grid from -L/2 to L/2
    t_grid = np.arange(0, ntime) * tau  # time grid 
    
    # Initializing psi
    psi = np.zeros((nspace, ntime), dtype=complex)
    psi[:, 0] = make_gaussIC(sigma0, k0, x0, x_i=x_grid)  # initial condition for Gaussian wavepacket at psi(x, t=0)

    # Iteratively solving for psi
    for n in range(1, ntime):
        psi[:,n] = np.matmul(A, psi[:,n-1])

    # Finding total probability
    # Conserved to within numerical error. Should technically be equal to 1 everywhere but we have approximated constants. 
    prob = np.zeros(ntime, dtype = complex)
    prob = np.sum(psi * np.conjugate(psi), axis=0) # sum all rows in each column, ie. total probability at each time step.
    
    return psi, x_grid, t_grid, prob



def sch_plot(sch_arrays, time = 0, type = 'psi', plotshow = True, save = False):
    '''
    Plots output of sch_eqn().

    Args:
    sch_arrays (1x4 tuple): tuple of sch_eqn outputs (psi, x_grid, t_grid, prob) 
    time (float-like): time (s) at which plot is desired. May be approximated.
    type (str): type of plot, either 'psi' (plot of the real part of ψ(x) at time t)
        or 'prob' (plot of the particle probability density ψ ψ*(x) at a specific time). Defaults to psi
    plotshow (bool): Option to display the plot. Defaults to True
    save (bool): Option to save to file. Defaults to False

    Returns:
    None
    '''
    psi = np.real(sch_arrays[0]) #taking real only
    x_grid = sch_arrays[1]
    t_grid = sch_arrays[2]
    prob = sch_arrays[0]*np.conjugate(sch_arrays[0])

    ### Match given time point to closest match in time grid, rounding down 
    t_index = np.searchsorted(t_grid, time, side = 'left') # left side means t_grid[i-1] < time <= t_grid[i], ie rounding down
    if t_index == len(t_grid):
        t_index = len(t_grid)
        raise ValueError("Given time outside solved range.")
    else:
        pass

    fig, ax = plt.subplots()
    t_label = np.round(t_grid[t_index], 2)
    # psi plot
    if type == 'psi':
        ax.plot(x_grid, psi[:, t_index])
        ax.set(xlabel = 'x', ylabel = f'psi(x, t={t_label})', title = f'Schroedinger Wavefunction Plot, t={t_label}')
    # prob plot
    elif type == 'prob':
        ax.plot(x_grid, prob[:, t_index])
        ax.set(xlabel = 'x', ylabel = f'P(x, t={t_label})', title = f'Schroedinger Wavefunction Probability Density Plot, t={t_label}')
    else:
        raise ValueError("Please choose either 'psi' or 'prob' as type.")
    

    if plotshow == True:
        plt.show()
    else:
        pass

    if save == True:
        savename = input("Please input name to save plot as, alphanumeric chars only:    ")
        if savename.isalnum() == True:
            fig.savefig(f"{savename}.png")
        else:
            raise ValueError("Please input alphanumeric chars only.")
    else:
        pass

    return



### sch_eqn examples

# # Sample plot
# import matplotlib.pyplot as plt
# psi, x_grid, _, _ = sch_eqn(nspace = 200, ntime = 500, tau = 1.0, method = 'crank')
# plt.plot(x_grid, psi[:, 0])
# plt.title('Psi(x) at t=0, Crank')
# plt.ylabel('Psi(x)')
# plt.xlabel('x')
# plt.show()

# # Sample animation
# import matplotlib.pyplot as plt
# from matplotlib import animation as animation

# fig, ax2 = plt.subplots(2, 1, figsize = (8, 16))
# psi, x_grid, _, _ = sch_eqn(nspace = 200, ntime = 500, tau = 1.0, method = 'crank', potential = [0])
# probability = psi*np.conjugate(psi) # Probability density calculation

# ax2[0].set(title = 'Psi(x)')
# ax2[1].set(title = 'Probability Density Function')

# psi_plot, = ax2[0].plot(x_grid, psi[:, 0])
# prob_plot, = ax2[1].plot(x_grid, probability[:, 0])

# def update(frame):
#     psi_plot.set_ydata(psi[:, frame])
#     prob_plot.set_ydata(probability[:, frame])
#     return (psi_plot, prob_plot)

# ani = animation.FuncAnimation(fig=fig, func=update, frames=500, interval=1)
# plt.show()


### sch_plot examples

# # Sample call
# sch_plot(sch_arrays=sch_eqn(nspace = 200, ntime = 500, tau = 1.0, method = 'crank'), time = 20, type = 'psi')
# sch_plot(sch_arrays=sch_eqn(nspace = 200, ntime = 500, tau = 1.0, method = 'crank'), time = 20, type = 'prob')


