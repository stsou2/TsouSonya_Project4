import numpy as np
import matplotlib.pyplot as plt


x_i =  np.linspace(-50, 50, 200, endpoint=False)
sigma_0, x_0, k_0 = 10, 0, 0.5
y = (1/(np.sqrt(sigma_0*np.sqrt(np.pi))))*(np.exp(1j*k_0*x_i))*np.exp((-(x_i-x_0)**2)/(2*sigma_0**2))



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

y2 = make_gaussIC(sigma_0, x_0, k_0, x_i)
plt.plot(x_i, y, 'b')
plt.plot(x_i, y2, 'm')


plt.show()