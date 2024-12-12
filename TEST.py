import numpy as np
import matplotlib.pyplot as plt


x_i =  np.linspace(-50, 50, 200, endpoint=False)
sigma_0, x_0, k_0 = 10, 0, 0.5
y = (1/(np.sqrt(sigma_0*np.sqrt(np.pi))))*(np.exp(1j*k_0*x_i))*np.exp((-(x_i-x_0)**2)/(2*sigma_0**2))
plt.plot(x_i, y)

plt.show()