import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

plt.rcParams['figure.figsize'] = [10, 10]
plt.rcParams['font.size'] = 12
plt.rcParams['mathtext.fontset'] = 'cm'

e_val = np.e
c_bound = e_val - 1

def interior_system(X, t):
    s, c = X
    ds = s - c
    dc = c + 1
    return [ds, dc]

def interior_system_backward(X, t):
    s, c = X
    ds = -(s - c)
    dc = -(c + 1)
    return [ds, dc]

fig, ax = plt.subplots(figsize=(10, 10))

s_min, s_max = -1, 6
c_min, c_max = -0.5, 6

s_grid = np.linspace(s_min, s_max, 20)
c_grid_interior = np.linspace(c_bound + 0.1, c_max, 12)
S_int, C_int = np.meshgrid(s_grid, c_grid_interior)
DS_int = S_int - C_int
DC_int = C_int + 1
mag_int = np.sqrt(DS_int**2 + DC_int**2)
mag_int[mag_int == 0] = 1
ax.quiver(S_int, C_int, DS_int/mag_int, DC_int/mag_int,
          mag_int, cmap='coolwarm', alpha=0.6, scale=25)

s_grid_bound = np.linspace(s_min, s_max, 20)
for s_b in s_grid_bound:
    ds_b = s_b - c_bound
    if abs(ds_b) > 0.05:
        arrow_len = 0.15 * np.sign(ds_b)
        ax.annotate('', xy=(s_b + arrow_len, c_bound),
                    xytext=(s_b, c_bound),
                    arrowprops=dict(arrowstyle='->',
                    color='purple', lw=1.2, alpha=0.6))

s_nc = np.linspace(s_min, s_max, 100)
ax.plot(s_nc, s_nc, 'b--', linewidth=2.5,
        label=r'$\dot{s} = 0$: $c = s$')
ax.axhline(y=c_bound, color='green', linestyle='-.',
           linewidth=2.5, label=r'Constraint: $c = e - 1$')

t_fw = np.linspace(0, 3, 500)
t_bw = np.linspace(0, 3, 500)

ics_interior = [
    (0.5, 2.5), (1, 3), (2, 2.5), (3, 2.0),
    (0, 4), (1, 4), (3, 3), (4, 2.5),
    (2, 4), (4, 4), (1.5, 2.0), (0.5, 3.5),
    (5, 3), (3, 5)
]

for s0, c0 in ics_interior:
    if c0 > c_bound:
        sol = odeint(interior_system, [s0, c0], t_fw)
        mask = ((sol[:,0] >= s_min) & (sol[:,0] <= s_max)
              & (sol[:,1] >= c_bound) & (sol[:,1] <= c_max))
        if np.any(mask):
            ax.plot(sol[mask,0], sol[mask,1],
                    'k-', lw=0.8, alpha=0.5)
            vi = np.where(mask)[0]
            if len(vi) > 10:
                mid = vi[len(vi)//4]
                if mid < len(sol)-1:
                    ax.annotate('',
                        xy=(sol[mid+1,0], sol[mid+1,1]),
                        xytext=(sol[mid,0], sol[mid,1]),
                        arrowprops=dict(arrowstyle='->',
                                        color='k', lw=1))
        sol_b = odeint(interior_system_backward,
                       [s0, c0], t_bw)
        mask_b = ((sol_b[:,0] >= s_min)
                & (sol_b[:,0] <= s_max)
                & (sol_b[:,1] >= c_bound)
                & (sol_b[:,1] <= c_max))
        if np.any(mask_b):
            ax.plot(sol_b[mask_b,0], sol_b[mask_b,1],
                    'k-', lw=0.8, alpha=0.5)

s_start = e_val - 1
c_start = e_val - 1

t_int = np.linspace(1.5, 2, 100)
c_opt = np.exp(t_int - 0.5) - 1
s_opt = (5 - 2*t_int) * np.exp(t_int - 0.5) / 2 - 1

ax.plot(s_start, c_start, 'go', markersize=12,
        markeredgecolor='darkgreen', markeredgewidth=2,
        zorder=10)
ax.plot(s_opt, c_opt, 'g-', linewidth=3.5,
        label='Optimal path', zorder=5)
ax.plot(s_opt[-1], c_opt[-1], 'gs', markersize=12,
        markeredgecolor='darkgreen', markeredgewidth=2,
        zorder=10)

mid = len(t_int) // 2
ax.annotate('', xy=(s_opt[mid+3], c_opt[mid+3]),
           xytext=(s_opt[mid], c_opt[mid]),
           arrowprops=dict(arrowstyle='->',
                           color='green', lw=2.5))

ax.annotate(r'$t=0$ to $t=3/2$' + '\n' + r'$(e-1,\, e-1)$',
           xy=(s_start, c_start),
           xytext=(s_start - 1.2, c_start + 0.8),
           fontsize=11, ha='center',
           arrowprops=dict(arrowstyle='->',
                           color='darkgreen', lw=1))

ax.text(4.5, 5.5, r'$\dot{s} < 0,\ \dot{c} > 0$',
        fontsize=11, ha='center',
        bbox=dict(fc='white', ec='gray', alpha=0.8))
ax.text(4.5, 2.5, r'$\dot{s} > 0,\ \dot{c} > 0$',
        fontsize=11, ha='center',
        bbox=dict(fc='white', ec='gray', alpha=0.8))

ax.set_xlim(s_min, s_max)
ax.set_ylim(c_min, c_max)
ax.set_xlabel(r'$s$ (state)', fontsize=16)
ax.set_ylabel(r'$c$ (control)', fontsize=16)
ax.set_title(
    r'Phase Diagram: $\dot{s} = s - c$, '
    r'$\dot{c} = c + 1$ (interior)'
    + '\n' + r'with constraint $c \geq e - 1$',
    fontsize=14)
ax.legend(loc='upper left', fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('phase_diagram_p3.png',
            dpi=300, bbox_inches='tight')
plt.show()
