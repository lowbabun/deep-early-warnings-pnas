"""May's harvesting model ramped through a fold, with early warning signals (EWS).

dV/dt = r V (1 - V/K) - 0.1 h V^2 / (V0^2 + V^2) + sigma xi(t)
"""

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess

# Model parameters
r = 0.333
K = 25.0
V0 = 3.0

# Simulation parameters
sigma = 0.3  # noise amplitude
hstart = 15.0
hend = 24.0
tmax = 1000.0
seed = 1  # set to None for a new noise realisation
dt = 0.01
dt_out = 1.0  # sampling interval of the recorded series

# EWS parameters (fractions of the pre-EWS data)
lowess_span = 0.2
rolling_window = 0.5
ews_end = None  # None -> bifurcation time

# Residual export (covers data up to ews_end, i.e. before the tipping point)
export_residuals = True
residuals_file = "residuals_pre_tipping.csv"


def f(V, h):
    return r * V * (1 - V / K) - 0.1 * h * V**2 / (V0**2 + V**2)


def h_eq(V):
    """Harvesting rate at which V is an equilibrium."""
    return 10 * r * (1 - V / K) * (V**2 + V0**2) / V


def folds():
    """Fold points (h, V): local extrema of h_eq(V)."""
    V = np.linspace(0.01, K - 0.01, 400001)
    hv = h_eq(V)
    d = np.diff(hv)
    idx = np.where(np.sign(d[:-1]) != np.sign(d[1:]))[0] + 1
    return [(hv[i], V[i]) for i in idx]


def upper_equilibrium(h):
    V = K
    for _ in range(200000):
        V += f(V, h) * 0.01
    return V


# Simulate (Euler-Maruyama)
rng = np.random.default_rng(seed)
n = int(tmax / dt)
stride = int(round(dt_out / dt))
t = np.arange(n) * dt
h = hstart + (hend - hstart) * t / tmax
V = np.empty(n)
V[0] = upper_equilibrium(hstart)
noise = rng.standard_normal(n - 1) * sigma * np.sqrt(dt)
for i in range(n - 1):
    V[i + 1] = max(V[i] + f(V[i], h[i]) * dt + noise[i], 0.0)

t_s, V_s, h_s = t[::stride], V[::stride], h[::stride]

# Bifurcation time: ramp reaches the upper fold
fold_list = folds()
h_fold, V_fold = max(fold_list)
t_bif = tmax * (h_fold - hstart) / (hend - hstart)
print("Fold bifurcations:", ", ".join(f"h = {a:.2f} (V = {b:.2f})" for a, b in fold_list))
print(f"Ramp crosses upper fold at t ~ {t_bif:.0f}")

# EWS on data up to ews_end
t_end = t_bif if ews_end is None else ews_end
m = t_s <= t_end
tp, Vp = t_s[m], V_s[m]
span_t = lowess_span * (tp[-1] - tp[0])
win_t = rolling_window * (tp[-1] - tp[0])
print(f"Lowess span ~ {span_t:.0f} time units, rolling window ~ {win_t:.0f} time units")

trend = lowess(Vp, tp, frac=lowess_span, return_sorted=False)
resid = Vp - trend

# Export residuals prior to the tipping point (time, V, trend, residual)``
if export_residuals:
    np.savetxt(
        residuals_file,
        np.column_stack([tp, Vp, trend, resid]),
        delimiter=",",
        header="time,V,trend,residual",
        comments="",
    )
    print(f"Saved {len(resid)} residuals (t <= {t_end:.0f}) to {residuals_file}")

w = int(round(rolling_window * len(Vp)))
variance = np.full(len(Vp), np.nan)
ac1 = np.full(len(Vp), np.nan)
for i in range(w, len(Vp) + 1):
    seg = resid[i - w:i]
    variance[i - 1] = seg.var()
    ac1[i - 1] = np.corrcoef(seg[:-1], seg[1:])[0, 1]

# Plots
fig, ax = plt.subplots(4, 1, figsize=(9, 10), sharex=True)
ax[0].plot(t_s, V_s, lw=0.8, label="V")
ax[0].plot(tp, trend, "r", lw=1.5, label="lowess trend")
ax[0].axvline(t_bif, color="k", ls="--", lw=0.8)
ax[0].set_ylabel("V")
ax[0].legend()
ax[1].plot(tp, resid, lw=0.6)
ax[1].set_ylabel("residuals")
ax[2].plot(tp, variance)
ax[2].set_ylabel("variance")
ax[3].plot(tp, ac1)
ax[3].set_ylabel("lag-1 autocorr")
ax[3].set_xlabel("time")
for a in ax[1:]:
    a.axvline(t_bif, color="k", ls="--", lw=0.8)
plt.tight_layout()
plt.show()
