import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import os

plt.rcParams['font.size'] = 10
plt.rcParams['mathtext.fontset'] = 'cm'

os.makedirs('sensitivity_output', exist_ok=True)

# ── Core functions ──────────────────────────────────────────────
def g(s):       return s * (1 - s)
def g_prime(s): return 1 - 2 * s

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


def plot_sc(ax, p_val, delta_val):
    """Plot (s,c)-phase diagram on a given axis."""
    s_hat = (1 - delta_val) / 2
    c_hat = g(s_hat)
    psi_hat = p_val - c_hat

    # Check feasibility
    if s_hat <= 0 or s_hat >= 1 or c_hat <= 0 or psi_hat <= 0:
        ax.text(0.5, 0.5, 'Infeasible\nparameters', ha='center', va='center',
                fontsize=12, transform=ax.transAxes)
        ax.set_xlim(0, 1); ax.set_ylim(0, 0.5)
        return

    # ODE systems
    def sc_fwd(t, X):
        s, c = X
        return [g(s) - c, -(p_val - c) * (delta_val - g_prime(s))]
    def sc_bwd(t, X):
        s, c = X
        return [-(g(s) - c), (p_val - c) * (delta_val - g_prime(s))]

    bounds = [0, 1, 0, 0.5]

    # Vector field
    S, C = np.meshgrid(np.linspace(0.02, 0.98, 16), np.linspace(0.02, 0.48, 12))
    DS = g(S) - C
    DC = -(p_val - C) * (delta_val - g_prime(S))
    mag = np.sqrt(DS**2 + DC**2); mag[mag == 0] = 1
    ax.quiver(S, C, DS/mag, DC/mag, mag, cmap='coolwarm', alpha=0.45, scale=30)

    # Nullclines
    s_nc = np.linspace(0, 1, 200)
    ax.plot(s_nc, g(s_nc), 'b--', lw=2, label=r'$\dot{s}=0$')
    ax.axvline(x=s_hat, color='r', ls='--', lw=2, label=r'$\dot{c}=0$')

    # Saddle paths via eigenvector
    J = np.array([[g_prime(s_hat), -1], [-2 * psi_hat, 0]])
    eigvals, eigvecs = np.linalg.eig(J)
    sv = eigvecs[:, np.argmin(eigvals)]

    eps = 1e-5
    for sign in [1, -1]:
        s0 = s_hat + sign * eps * sv[0]
        c0 = c_hat + sign * eps * sv[1]
        sx, cx = integrate_path(sc_bwd, [s0, c0], [0, 200], bounds)
        ax.plot(sx, cx, 'g-', lw=2.5, zorder=5,
                label='Saddle path' if sign == 1 else '')

    # General trajectories
    for s0, c0 in [(0.1,0.35),(0.2,0.4),(0.8,0.4),(0.9,0.35),
                    (0.15,0.05),(0.85,0.05),(0.3,0.38),(0.7,0.38)]:
        sx, cx = integrate_path(sc_fwd, [s0, c0], [0, 10], bounds, max_step=0.02)
        ax.plot(sx, cx, 'k-', lw=0.6, alpha=0.35)

    # Equilibrium dot
    ax.plot(s_hat, c_hat, 'ko', ms=7, markerfacecolor='yellow',
            markeredgewidth=1.5, zorder=10)
    ax.annotate(f'$\\hat{{s}}={s_hat:.2f}$\n$\\hat{{c}}={c_hat:.3f}$',
                xy=(s_hat, c_hat), xytext=(s_hat + 0.1, c_hat + 0.07),
                fontsize=8, bbox=dict(boxstyle='round', fc='lightyellow',
                                       ec='orange', alpha=0.85))

    ax.set_xlim(0, 1); ax.set_ylim(0, 0.5)
    ax.set_xlabel(r'$s$', fontsize=11)
    ax.set_ylabel(r'$c$', fontsize=11)
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, alpha=0.25)


def plot_spsi(ax, p_val, delta_val):
    """Plot (s,psi)-phase diagram on a given axis."""
    s_hat = (1 - delta_val) / 2
    c_hat = g(s_hat)
    psi_hat = p_val - c_hat

    if s_hat <= 0 or s_hat >= 1 or c_hat <= 0 or psi_hat <= 0:
        ax.text(0.5, 0.5, 'Infeasible\nparameters', ha='center', va='center',
                fontsize=12, transform=ax.transAxes)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.2)
        return

    def spsi_fwd(t, X):
        s, psi = X
        return [g(s) - (p_val - psi), psi * (delta_val - g_prime(s))]
    def spsi_bwd(t, X):
        s, psi = X
        return [-(g(s) - (p_val - psi)), -(psi * (delta_val - g_prime(s)))]

    psi_max = max(1.2, psi_hat + 0.5)
    bounds = [0, 1, 0, psi_max]

    # Vector field
    S, PSI = np.meshgrid(np.linspace(0.02, 0.98, 16),
                         np.linspace(0.05, psi_max - 0.05, 12))
    DS = g(S) - (p_val - PSI)
    DPSI = PSI * (delta_val - g_prime(S))
    mag = np.sqrt(DS**2 + DPSI**2); mag[mag == 0] = 1
    ax.quiver(S, PSI, DS/mag, DPSI/mag, mag, cmap='coolwarm', alpha=0.45, scale=30)

    # Nullclines
    s_nc = np.linspace(0.01, 0.99, 200)
    ax.plot(s_nc, p_val - g(s_nc), 'b--', lw=2, label=r'$\dot{s}=0$')
    ax.axvline(x=s_hat, color='r', ls='--', lw=2, label=r'$\dot{\psi}=0$')

    # Saddle paths
    J = np.array([[g_prime(s_hat), 1], [2 * psi_hat, 0]])
    eigvals, eigvecs = np.linalg.eig(J)
    sv = eigvecs[:, np.argmin(eigvals)]

    eps = 1e-5
    for sign in [1, -1]:
        s0 = s_hat + sign * eps * sv[0]
        psi0 = psi_hat + sign * eps * sv[1]
        sx, psix = integrate_path(spsi_bwd, [s0, psi0], [0, 200], bounds)
        ax.plot(sx, psix, 'g-', lw=2.5, zorder=5,
                label='Saddle path' if sign == 1 else '')

    # General trajectories
    frac = 0.7
    for s0, psi_frac in [(0.1,0.8),(0.2,0.65),(0.8,0.65),(0.9,0.8),
                          (0.15,1.1),(0.85,1.1),(0.3,1.0),(0.7,1.0)]:
        psi0 = min(psi_frac, psi_max - 0.05)
        sx, psix = integrate_path(spsi_fwd, [s0, psi0], [0, 10], bounds, max_step=0.02)
        ax.plot(sx, psix, 'k-', lw=0.6, alpha=0.35)

    # Equilibrium
    ax.plot(s_hat, psi_hat, 'ko', ms=7, markerfacecolor='yellow',
            markeredgewidth=1.5, zorder=10)
    ax.annotate(f'$\\hat{{s}}={s_hat:.2f}$\n$\\hat{{\\psi}}={psi_hat:.3f}$',
                xy=(s_hat, psi_hat), xytext=(s_hat + 0.1, psi_hat + 0.08),
                fontsize=8, bbox=dict(boxstyle='round', fc='lightyellow',
                                       ec='orange', alpha=0.85))

    ax.set_xlim(0, 1); ax.set_ylim(0, psi_max)
    ax.set_xlabel(r'$s$', fontsize=11)
    ax.set_ylabel(r'$\psi$', fontsize=11)
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, alpha=0.25)


# ══════════════════════════════════════════════════════════════════
# PAGE 1: Varying delta (discount rate), fixed p = 1
# ══════════════════════════════════════════════════════════════════
deltas = [0.01, 0.1, 0.3, 0.5]
p_fixed = 1.0

fig1, axes1 = plt.subplots(2, 2, figsize=(12, 10))
fig1.suptitle(r'Sensitivity: varying $\delta$ (discount rate), $p = 1$ fixed'
              '\n' + r'$(s, c)$-phase diagrams', fontsize=14, fontweight='bold')

for ax, d in zip(axes1.flat, deltas):
    s_eq = (1 - d) / 2
    ax.set_title(rf'$\delta = {d}$ $\longrightarrow$ $\hat{{s}} = {s_eq:.3f}$', fontsize=11)
    plot_sc(ax, p_fixed, d)

fig1.tight_layout(rect=[0, 0, 1, 0.93])
fig1.savefig('sensitivity_output/sensitivity_delta_sc.png', dpi=300, bbox_inches='tight')
print("Saved sensitivity_delta_sc.png")


fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10))
fig2.suptitle(r'Sensitivity: varying $\delta$ (discount rate), $p = 1$ fixed'
              '\n' + r'$(s, \psi)$-phase diagrams', fontsize=14, fontweight='bold')

for ax, d in zip(axes2.flat, deltas):
    s_eq = (1 - d) / 2
    ax.set_title(rf'$\delta = {d}$ $\longrightarrow$ $\hat{{s}} = {s_eq:.3f}$', fontsize=11)
    plot_spsi(ax, p_fixed, d)

fig2.tight_layout(rect=[0, 0, 1, 0.93])
fig2.savefig('sensitivity_output/sensitivity_delta_spsi.png', dpi=300, bbox_inches='tight')
print("Saved sensitivity_delta_spsi.png")


# ══════════════════════════════════════════════════════════════════
# PAGE 2: Varying p (fish price), fixed delta = 0.1
# ══════════════════════════════════════════════════════════════════
prices = [0.5, 1.0, 1.5, 2.0]
delta_fixed = 0.1

fig3, axes3 = plt.subplots(2, 2, figsize=(12, 10))
fig3.suptitle(r'Sensitivity: varying $p$ (fish price), $\delta = 0.1$ fixed'
              '\n' + r'$(s, c)$-phase diagrams', fontsize=14, fontweight='bold')

for ax, pv in zip(axes3.flat, prices):
    s_eq = (1 - delta_fixed) / 2
    psi_eq = pv - g(s_eq)
    ax.set_title(rf'$p = {pv}$ $\longrightarrow$ $\hat{{\psi}} = {psi_eq:.3f}$', fontsize=11)
    plot_sc(ax, pv, delta_fixed)

fig3.tight_layout(rect=[0, 0, 1, 0.93])
fig3.savefig('sensitivity_output/sensitivity_price_sc.png', dpi=300, bbox_inches='tight')
print("Saved sensitivity_price_sc.png")


fig4, axes4 = plt.subplots(2, 2, figsize=(12, 10))
fig4.suptitle(r'Sensitivity: varying $p$ (fish price), $\delta = 0.1$ fixed'
              '\n' + r'$(s, \psi)$-phase diagrams', fontsize=14, fontweight='bold')

for ax, pv in zip(axes4.flat, prices):
    s_eq = (1 - delta_fixed) / 2
    psi_eq = pv - g(s_eq)
    ax.set_title(rf'$p = {pv}$ $\longrightarrow$ $\hat{{\psi}} = {psi_eq:.3f}$', fontsize=11)
    plot_spsi(ax, pv, delta_fixed)

fig4.tight_layout(rect=[0, 0, 1, 0.93])
fig4.savefig('sensitivity_output/sensitivity_price_spsi.png', dpi=300, bbox_inches='tight')
print("Saved sensitivity_price_spsi.png")

print("\nDone! All 4 figures (16 phase diagrams) saved to sensitivity_output/")
plt.show()
