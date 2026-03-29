import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

plt.rcParams['figure.figsize'] = [10, 10]
plt.rcParams['font.size'] = 12
plt.rcParams['mathtext.fontset'] = 'cm'

# Parameters
p = 1.0
delta = 0.1

def g(s):
    return s * (1 - s)

def g_prime(s):
    return 1 - 2 * s

# Equilibrium
s_hat = (1 - delta) / 2  # 0.45
c_hat = g(s_hat)          # 0.2475
psi_hat = p - c_hat        # 0.7525

print(f"Equilibrium: s_hat={s_hat}, c_hat={c_hat:.4f}, psi_hat={psi_hat:.4f}")


def integrate_path(system, x0, t_span, bounds, max_step=0.01):
    s_lo, s_hi, y_lo, y_hi = bounds
    def event_out(t, X):
        s, y = X
        margin = 0.001
        return min(s - s_lo + margin, s_hi - s + margin,
                   y - y_lo + margin, y_hi - y + margin)
    event_out.terminal = True
    sol = solve_ivp(system, t_span, x0,
                    max_step=max_step, events=event_out,
                    dense_output=True, rtol=1e-10, atol=1e-12)
    return sol.y[0], sol.y[1]


# Jacobian for (s,c) system
J_sc = np.array([[g_prime(s_hat), -1],
                 [-2 * psi_hat, 0]])
eigvals_sc, eigvecs_sc = np.linalg.eig(J_sc)
print(f"(s,c) eigenvalues: {eigvals_sc}")

neg_idx = 0 if eigvals_sc[0] < 0 else 1
pos_idx = 1 - neg_idx
stable_vec_sc = eigvecs_sc[:, neg_idx]
unstable_vec_sc = eigvecs_sc[:, pos_idx]


# =============================================
# FIGURE 1: (s, c)-phase diagram
# =============================================

def sc_forward(t, X):
    s, c = X
    ds = g(s) - c
    psi = p - c
    dc = -psi * (delta - g_prime(s))
    return [ds, dc]

def sc_backward(t, X):
    s, c = X
    ds = -(g(s) - c)
    psi = p - c
    dc = psi * (delta - g_prime(s))
    return [ds, dc]

fig1, ax1 = plt.subplots(figsize=(10, 8))

s_min, s_max = 0, 1
c_min, c_max = 0, 0.5
bounds_sc = [s_min, s_max, c_min, c_max]

# Vector field
s_grid = np.linspace(0.02, 0.98, 18)
c_grid = np.linspace(0.02, 0.48, 14)
S, C = np.meshgrid(s_grid, c_grid)
DS = g(S) - C
PSI = p - C
DC = -(PSI) * (delta - g_prime(S))
mag = np.sqrt(DS**2 + DC**2)
mag[mag == 0] = 1
ax1.quiver(S, C, DS/mag, DC/mag, mag,
           cmap='coolwarm', alpha=0.5, scale=28)

# Nullclines
s_nc = np.linspace(0, 1, 200)
ax1.plot(s_nc, g(s_nc), 'b--', linewidth=2.5,
         label=r'$\dot{s}=0$: $c = s(1-s)$')
ax1.axvline(x=s_hat, color='r', linestyle='--',
            linewidth=2.5,
            label=r"$\dot{c}=0$: $g'(s)=\delta$, $s=0.45$")

# Saddle paths: perturb along stable eigenvector, integrate backward
eps = 1e-5
for sign in [1, -1]:
    s0 = s_hat + sign * eps * stable_vec_sc[0]
    c0 = c_hat + sign * eps * stable_vec_sc[1]
    sx, cx = integrate_path(sc_backward, [s0, c0],
                            [0, 200], bounds_sc)
    ax1.plot(sx, cx, 'g-', linewidth=3, zorder=5,
             label='Saddle path (optimal)' if sign == 1 else '')

# General trajectories
ics_sc = [
    (0.1, 0.35), (0.2, 0.40), (0.8, 0.40),
    (0.9, 0.35), (0.15, 0.05), (0.85, 0.05),
    (0.3, 0.38), (0.7, 0.38),
    (0.5, 0.10), (0.3, 0.10),
]
for s0, c0 in ics_sc:
    sx, cx = integrate_path(sc_forward, [s0, c0],
                            [0, 10], bounds_sc, max_step=0.02)
    ax1.plot(sx, cx, 'k-', lw=0.7, alpha=0.4)
    n = len(sx)
    if n > 10:
        mid = n // 5
        dx = sx[mid+1] - sx[mid]
        dy = cx[mid+1] - cx[mid]
        ax1.annotate('', xy=(sx[mid]+dx*0.5, cx[mid]+dy*0.5),
                    xytext=(sx[mid], cx[mid]),
                    arrowprops=dict(arrowstyle='->', color='k', lw=0.8))

# Equilibrium
ax1.plot(s_hat, c_hat, 'ko', markersize=10,
         markerfacecolor='yellow', markeredgewidth=2, zorder=10)
ax1.annotate(f'$({s_hat}, {c_hat:.4f})$\nSaddle point',
            xy=(s_hat, c_hat),
            xytext=(s_hat + 0.12, c_hat + 0.08),
            fontsize=11,
            bbox=dict(boxstyle='round', fc='lightyellow',
                      ec='orange', alpha=0.9))

# Direction labels
ax1.text(0.15, 0.42, r'$\dot{s}<0,\ \dot{c}>0$', fontsize=10,
         bbox=dict(fc='white', ec='gray', alpha=0.8))
ax1.text(0.65, 0.42, r'$\dot{s}<0,\ \dot{c}<0$', fontsize=10,
         bbox=dict(fc='white', ec='gray', alpha=0.8))
ax1.text(0.15, 0.05, r'$\dot{s}>0,\ \dot{c}>0$', fontsize=10,
         bbox=dict(fc='white', ec='gray', alpha=0.8))
ax1.text(0.65, 0.05, r'$\dot{s}>0,\ \dot{c}<0$', fontsize=10,
         bbox=dict(fc='white', ec='gray', alpha=0.8))

ax1.set_xlim(s_min, s_max)
ax1.set_ylim(c_min, c_max)
ax1.set_xlabel(r'$s$ (fish stock)', fontsize=16)
ax1.set_ylabel(r'$c$ (catch rate)', fontsize=16)
ax1.set_title(r'$(s, c)$-Phase Diagram: $\dot{s}=s(1-s)-c$, '
              + r'$\dot{c}=-(1-c)(2s-0.9)$'
              + f'\n($p={p}$, $\\delta={delta}$)',
              fontsize=13)
handles, labels = ax1.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax1.legend(by_label.values(), by_label.keys(),
           loc='upper right', fontsize=10)
ax1.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('phase_diagram_sc.png', dpi=300, bbox_inches='tight')
print("Saved phase_diagram_sc.png")


# =============================================
# FIGURE 2: (s, psi)-phase diagram
# =============================================

J_spsi = np.array([[g_prime(s_hat), 1],
                   [2 * psi_hat, 0]])
eigvals_spsi, eigvecs_spsi = np.linalg.eig(J_spsi)
print(f"\n(s,psi) eigenvalues: {eigvals_spsi}")

neg_idx2 = 0 if eigvals_spsi[0] < 0 else 1
pos_idx2 = 1 - neg_idx2
stable_vec_spsi = eigvecs_spsi[:, neg_idx2]
unstable_vec_spsi = eigvecs_spsi[:, pos_idx2]


def spsi_forward(t, X):
    s, psi = X
    c = p - psi
    ds = g(s) - c
    dpsi = psi * (delta - g_prime(s))
    return [ds, dpsi]

def spsi_backward(t, X):
    s, psi = X
    c = p - psi
    ds = -(g(s) - c)
    dpsi = -(psi * (delta - g_prime(s)))
    return [ds, dpsi]

fig2, ax2 = plt.subplots(figsize=(10, 8))

psi_min, psi_max = 0, 1.2
bounds_spsi = [s_min, s_max, psi_min, psi_max]

# Vector field
s_grid2 = np.linspace(0.02, 0.98, 18)
psi_grid2 = np.linspace(0.05, 1.15, 14)
S2, PSI2 = np.meshgrid(s_grid2, psi_grid2)
C2 = p - PSI2
DS2 = g(S2) - C2
DPSI2 = PSI2 * (delta - g_prime(S2))
mag2 = np.sqrt(DS2**2 + DPSI2**2)
mag2[mag2 == 0] = 1
ax2.quiver(S2, PSI2, DS2/mag2, DPSI2/mag2, mag2,
           cmap='coolwarm', alpha=0.5, scale=28)

# Nullclines
s_nc2 = np.linspace(0.01, 0.99, 200)
psi_nc = p - g(s_nc2)
ax2.plot(s_nc2, psi_nc, 'b--', linewidth=2.5,
         label=r'$\dot{s}=0$: $\psi = 1 - s(1-s)$')
ax2.axvline(x=s_hat, color='r', linestyle='--',
            linewidth=2.5,
            label=r"$\dot{\psi}=0$: $g'(s)=\delta$, $s=0.45$")

# Saddle paths
eps2 = 1e-5
for sign in [1, -1]:
    s0 = s_hat + sign * eps2 * stable_vec_spsi[0]
    psi0 = psi_hat + sign * eps2 * stable_vec_spsi[1]
    sx, psix = integrate_path(spsi_backward, [s0, psi0],
                              [0, 200], bounds_spsi)
    ax2.plot(sx, psix, 'g-', linewidth=3, zorder=5,
             label='Saddle path (optimal)' if sign == 1 else '')

# General trajectories
ics_spsi = [
    (0.1, 0.6), (0.2, 0.5), (0.8, 0.5),
    (0.9, 0.6), (0.15, 1.0), (0.85, 1.0),
    (0.3, 0.9), (0.7, 0.9),
]
for s0, psi0 in ics_spsi:
    sx, psix = integrate_path(spsi_forward, [s0, psi0],
                              [0, 10], bounds_spsi, max_step=0.02)
    ax2.plot(sx, psix, 'k-', lw=0.7, alpha=0.4)
    n = len(sx)
    if n > 10:
        mid = n // 5
        dx = sx[mid+1] - sx[mid]
        dy = psix[mid+1] - psix[mid]
        ax2.annotate('', xy=(sx[mid]+dx*0.5, psix[mid]+dy*0.5),
                    xytext=(sx[mid], psix[mid]),
                    arrowprops=dict(arrowstyle='->', color='k', lw=0.8))

# Equilibrium
ax2.plot(s_hat, psi_hat, 'ko', markersize=10,
         markerfacecolor='yellow', markeredgewidth=2, zorder=10)
ax2.annotate(f'$({s_hat}, {psi_hat:.4f})$\nSaddle point',
            xy=(s_hat, psi_hat),
            xytext=(s_hat + 0.12, psi_hat + 0.1),
            fontsize=11,
            bbox=dict(boxstyle='round', fc='lightyellow',
                      ec='orange', alpha=0.9))

# Direction labels
ax2.text(0.15, 1.05, r'$\dot{s}>0,\ \dot{\psi}<0$', fontsize=10,
         bbox=dict(fc='white', ec='gray', alpha=0.8))
ax2.text(0.65, 1.05, r'$\dot{s}>0,\ \dot{\psi}>0$', fontsize=10,
         bbox=dict(fc='white', ec='gray', alpha=0.8))
ax2.text(0.15, 0.55, r'$\dot{s}<0,\ \dot{\psi}<0$', fontsize=10,
         bbox=dict(fc='white', ec='gray', alpha=0.8))
ax2.text(0.65, 0.55, r'$\dot{s}<0,\ \dot{\psi}>0$', fontsize=10,
         bbox=dict(fc='white', ec='gray', alpha=0.8))

ax2.set_xlim(s_min, s_max)
ax2.set_ylim(psi_min, psi_max)
ax2.set_xlabel(r'$s$ (fish stock)', fontsize=16)
ax2.set_ylabel(r'$\psi$ (shadow price)', fontsize=16)
ax2.set_title(r'$(s, \psi)$-Phase Diagram: $\dot{s}=s(1-s)-(1-\psi)$, '
              + r'$\dot{\psi}=\psi(2s-0.9)$'
              + f'\n($p={p}$, $\\delta={delta}$)',
              fontsize=13)
handles2, labels2 = ax2.get_legend_handles_labels()
by_label2 = dict(zip(labels2, handles2))
ax2.legend(by_label2.values(), by_label2.keys(),
           loc='upper right', fontsize=10)
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('phase_diagram_spsi.png', dpi=300, bbox_inches='tight')
print("Saved phase_diagram_spsi.png")

plt.show()
