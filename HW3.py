import numpy as np
import pyshtools as pysh
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'figure.titlesize': 18,
})

def plot_relative_error(theta, rel_error, N, ax=None, use_log=True):
    """
    Plot relative error of the Legendre-function normalization sum
    against co-latitude, for a single value of Nmax.
    """
    if ax is None:
        _, ax = plt.subplots()
 
    theta_deg = np.rad2deg(theta)
 
    if use_log:
        ax.semilogy(theta_deg, np.abs(rel_error), lw=1.2)
        ax.set_ylabel('|Relative error|')
    else:
        ax.plot(theta_deg, rel_error, lw=1.2)
        ax.set_ylabel('Relative error')
 
    ax.set_xlabel(r'Co-latitude, $\theta$ (deg)')
    ax.set_title(f'Nmax = {N}')
    ax.set_xlim(theta_deg.min(), theta_deg.max())
    ax.grid(True, which='both', alpha=0.3)
 
    return ax

if __name__ == '__main__':
    theta = np.deg2rad(np.linspace(0.5, 179.5, 180))
    Nmax = np.array([100,500,1000])
    targets = (Nmax + 1)**2
    sums = np.zeros((len(theta),len(Nmax)))

    for i, N in enumerate(Nmax):
        sums[:,i] = np.array([np.sum(pysh.legendre.PlmBar(N,np.cos(t))**2) 
                            for t in theta])

    rel_error = (sums - targets) / targets
 
    for i, N in enumerate(Nmax):
        ax = plot_relative_error(theta, rel_error[:, i], N)
        ax.figure.suptitle(r'Relative error in $\sum_{l,m}\bar{P}_{lm}(\cos\theta)^2$ '
                            r'vs. $(N_{max}+1)^2$')

    plt.show()