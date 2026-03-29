import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from numpy.linalg import eig

# Parameters & Equilibrium
p, d = 1.0, 0.1
sh = (1 - d) / 2
ch = sh * (1 - sh)
ph = p - ch

# Define the ODE systems and calculate eigenvectors
def sc(t, y): return [y[0]*(1-y[0]) - y[1], -(p-y[1])*(d - (1-2*y[0]))]
def spsi(t, y): return [y[0]*(1-y[0]) - (p-y[1]), y[1]*(d - (1-2*y[0]))]

_, v1 = eig([[1-2*sh, -1], [-2*ph, 0]]); sv1 = v1[:, np.argmin(_)]
_, v2 = eig([[1-2*sh, 1], [2*ph, 0]]); sv2 = v2[:, np.argmin(_)]

# reusable plot 
def plot_phase(sys_func, y_eq, sv, ylim, nc_y, ics):
    plt.figure(figsize=(8, 6))

    # Nullclines, vector field, n trajectories
    s_vals = np.linspace(0, 1, 100)
    plt.plot(s_vals, nc_y(s_vals), 'b--', lw=2)
    plt.axvline(sh, color='r', ls='--', lw=2)
    plt.plot(sh, y_eq, 'ko', markersize=8, zorder=10)

    S, Y = np.meshgrid(np.linspace(0.02, 0.98, 18), np.linspace(ylim[0]+0.02, ylim[1]-0.02, 14))
    U, V = np.array(sys_func(0, [S, Y]))
    N = np.hypot(U, V)
    plt.quiver(S, Y, U/np.where(N==0, 1, N), V/np.where(N==0, 1, N), N, cmap='coolwarm', alpha=0.5)

    for ic in ics:
        sol = solve_ivp(sys_func, [0, 10], ic, max_step=0.05)
        plt.plot(sol.y[0], sol.y[1], 'k-', alpha=0.4)

    # Saddle Paths!
    for sign in [-1, 1]:
        sol = solve_ivp(lambda t, y: -np.array(sys_func(t, y)), [0, 100], 
                        [sh + sign*1e-5*sv[0], y_eq + sign*1e-5*sv[1]])
        plt.plot(sol.y[0], sol.y[1], 'g-', lw=2)

    plt.xlim(0, 1); plt.ylim(ylim)

# 1. plot (s, c)
plot_phase(sc, ch, sv1, (0, 0.5), lambda s: s*(1-s),
           [(0.1, 0.35), (0.2, 0.4), (0.8, 0.4), (0.9, 0.35), (0.15, 0.05), 
            (0.85, 0.05), (0.3, 0.38), (0.7, 0.38), (0.5, 0.1), (0.3, 0.1)])
plt.title("s, c Phase Diagram")

# 2. plot (s, psi)
plot_phase(spsi, ph, sv2, (0, 1.2), lambda s: p - s*(1-s),
           [(0.1, 0.6), (0.2, 0.5), (0.8, 0.5), (0.9, 0.6), (0.15, 1.0), 
            (0.85, 1.0), (0.3, 0.9), (0.7, 0.9)])
plt.title("s, psi Phase Diagram")

plt.show()