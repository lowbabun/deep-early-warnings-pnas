"""
Porpoising (F1 ground-effect bouncing) as a Hopf-bifurcation data generator
for a bifurcation-type classifier (Hopf vs Null, optional Fold).

Nondimensional ride-height model (3 states: height x, velocity v, aero force f):

    x' = v
    v' = k (x_r - x) - c v - f  + sigma * xi(t)        (xi = white noise: track roughness)
    tau f' = U^2 q(x) - f                                (first-order aero lag)
    q(x) = x * exp(1 - x)       (peaks at x = 1: "choke" height; q' < 0 above, > 0 below)

U (speed) is the bifurcation parameter and is ramped U(t) = U0 + r t.
  * Hopf mode: car sits on the choked side (x < 1). Dq/dx > 0 plus the aero lag
    gives an oscillatory instability (porpoising).
  * Fold mode (optional): large free height x_r, car on the un-choked side (x > 1),
    ground-suction fold (equilibrium disappears).
"""
import numpy as np
from scipy.optimize import brentq
from scipy.ndimage import gaussian_filter1d

# ----------------------------------------------------------------- model pieces
def q(x):  return x * np.exp(1.0 - x)
def dq(x): return np.exp(1.0 - x) * (1.0 - x)

def equilibrium(U, k, xr):
    g = lambda x: k * (xr - x) - U**2 * q(x)
    return brentq(g, 1e-3, xr)

def jacobian(U, k, c, tau, xr):
    x = equilibrium(U, k, xr)
    return np.array([[0, 1, 0],
                     [-k, -c, -1],
                     [U**2 * dq(x) / tau, 0, -1.0 / tau]])

def max_real(U, k, c, tau, xr):
    return np.linalg.eigvals(jacobian(U, k, c, tau, xr)).real.max()

def hopf_speed(k, c, tau, xr, Umax=6.0):
    """Smallest U at which the leading eigenvalue pair crosses Re=0 (None if none)."""
    Us = np.linspace(0.05, Umax, 600)
    m = np.array([max_real(U, k, c, tau, xr) for U in Us])
    idx = np.where((m[:-1] < 0) & (m[1:] >= 0))[0]
    if len(idx) == 0:
        return None
    i = idx[0]
    UH = brentq(lambda U: max_real(U, k, c, tau, xr), Us[i], Us[i + 1])
    ev = np.linalg.eigvals(jacobian(UH, k, c, tau, xr))
    if np.abs(ev.imag).max() < 1e-3:      # real crossing => not Hopf
        return None
    return UH

# ----------------------------------------------------------------- simulation
def simulate(U_path, k, c, tau, xr, sigma, dt=0.01, sub=50, seed=0, sigma_f=0.0, sigma_obs=0.0):
    """Euler-Maruyama along a prescribed speed path (one sample).
    U_path has one value per *recorded* point; returns the recorded x(t).
    White-noise channels (all Gaussian, uncorrelated in time):
      sigma      track roughness, forces the velocity equation (dynamic noise)
      sigma_f    flow turbulence, forces the aero-force equation (dynamic noise)
      sigma_obs  sensor noise added to the recorded height (observation noise,
                 does not feed back into the dynamics)"""
    rng = np.random.default_rng(seed)
    n = len(U_path)
    x = equilibrium(U_path[0], k, xr)
    v = 0.0
    f = U_path[0]**2 * q(x)
    out = np.empty(n)
    sq = sigma * np.sqrt(dt)
    sqf = sigma_f * np.sqrt(dt) / tau
    for i in range(n):
        U2 = U_path[i]**2
        for _ in range(sub):
            xc = max(x, 0.02)                         # ground contact floor
            dv = (k * (xr - x) - c * v - f) * dt + sq * rng.standard_normal()
            df = (U2 * q(xc) - f) / tau * dt + sqf * rng.standard_normal()
            x += v * dt
            v += dv
            f += df
        out[i] = x + sigma_obs * rng.standard_normal()
    return out

# White-noise levels, drawn uniformly per series from (low, high).
# Set a range to (0, 0) to switch that channel off.
NOISE = {
    "sigma":     (0.01, 0.05),   # track roughness on the velocity equation
    "sigma_f":   (0.00, 0.03),   # flow turbulence on the aero-force equation
    "sigma_obs": (0.00, 0.01),   # sensor noise on the recorded ride height
}

def make_sample(rng, label, L=500, n_pts=900, dt=0.01, sub=50, pad=False):
    """label: 'hopf' (speed ramped through the Hopf point) or 'null' (constant speed
    well below it). Returns recorded ride height x[0:n_pts] and the transition
    index iH (n_pts for null)."""
    while True:
        k   = rng.uniform(0.8, 1.2)
        c   = rng.uniform(0.2, 0.4)
        tau = rng.uniform(0.35, 0.7)
        xr  = rng.uniform(1.4, 1.8)
        UH  = hopf_speed(k, c, tau, xr)
        if UH is not None:
            break
    sigma     = rng.uniform(*NOISE["sigma"])
    sigma_f   = rng.uniform(*NOISE["sigma_f"])
    sigma_obs = rng.uniform(*NOISE["sigma_obs"])
    if label == "hopf":
        lo = 60 if pad else L + 20                       # earliest allowed crossing index
        iH = int(rng.integers(lo, n_pts - 100))
        U0 = rng.uniform(0.35, 0.6) * UH
        Uend = U0 + (UH - U0) * (n_pts - 1) / iH         # linear ramp that hits UH at iH
        U = np.linspace(U0, Uend, n_pts)
    else:
        U = np.full(n_pts, rng.uniform(0.35, 0.6) * UH)
        iH = n_pts
    x = simulate(U, k, c, tau, xr, sigma, dt, sub, seed=int(rng.integers(1 << 31)),
                 sigma_f=sigma_f, sigma_obs=sigma_obs)
    return x, iH

def to_window(x, iH, rng, L=500, pad=False, min_len=50, detrend=True, bw=0.1):
    """Cut a window ending before the transition, take residuals, normalise by mean |.|.
    pad=False -> every window is exactly L points (no zero padding)."""
    if pad:
        end = int(rng.integers(min(min_len, iH - 1), iH))
    else:
        end = int(rng.integers(L, iH))
    seg = x[max(0, end - L):end]
    if detrend:
        seg = seg - gaussian_filter1d(seg, sigma=max(2, bw * len(seg)), mode="nearest")
    m = np.mean(np.abs(seg))
    seg = seg / m if m > 0 else seg
    out = np.zeros(L)
    out[L - len(seg):] = seg
    return out

# ----------------------------------------------------------------- fast / batched generation
def hopf_speed_fast(k, c, tau, xr):
    """Closed-form Hopf speed (vectorised over arrays), no eigenvalue scan.
    Hopf condition (cubic's Routh-Hurwitz): G = U^2 q'(x*) = c/tau + c^2 + c k tau =: G_H.
    With the equilibrium k (xr - x) = U^2 q(x) and q/q' = x/(1-x) this reduces to
        k x^2 - (k (xr + 1) + G_H) x + k xr = 0,
    whose smaller root lies on the choked side (x < 1); then U_H = sqrt(G_H / q'(x))."""
    k, c, tau, xr = map(np.asarray, (k, c, tau, xr))
    GH = c / tau + c**2 + c * k * tau
    B = k * (xr + 1.0) + GH
    x = (B - np.sqrt(B**2 - 4.0 * k**2 * xr)) / (2.0 * k)
    return np.sqrt(GH / dq(x))


def simulate_batch(U_paths, k, c, tau, xr, sigma, sigma_f, sigma_obs,
                   dt=0.01, sub=50, rng=None):
    """Euler-Maruyama for N series at once. U_paths: (N, n_pts) speed at each recorded
    point; all parameter arguments are length-N arrays. Returns recorded x, (N, n_pts)."""
    rng = np.random.default_rng() if rng is None else rng
    N, n = U_paths.shape
    x = np.array([equilibrium(U_paths[i, 0], k[i], xr[i]) for i in range(N)])
    v = np.zeros(N)
    f = U_paths[:, 0] ** 2 * q(x)
    out = np.empty((N, n))
    sq, sqf = sigma * np.sqrt(dt), sigma_f * np.sqrt(dt) / tau
    for i in range(n):
        U2 = U_paths[:, i] ** 2
        noise = rng.standard_normal((sub, 2, N))
        for j in range(sub):
            xc = np.maximum(x, 0.02)                     # ground contact floor
            dv = (k * (xr - x) - c * v - f) * dt + sq * noise[j, 0]
            df = (U2 * q(xc) - f) / tau * dt + sqf * noise[j, 1]
            x = x + v * dt
            v = v + dv
            f = f + df
        out[:, i] = x + sigma_obs * rng.standard_normal(N)
    return out


def make_batch(rng, labels, L=500, n_pts=900, dt=0.01, sub=50, pad=False):
    """labels: list of 'hopf'/'null', one per series. Same model, parameter ranges and
    noise ranges as make_sample, generated together. Returns (records (N, n_pts), iH (N,))."""
    N = len(labels)
    is_h = np.array([lab == "hopf" for lab in labels])
    k, c = rng.uniform(0.8, 1.2, N), rng.uniform(0.2, 0.4, N)
    tau, xr = rng.uniform(0.35, 0.7, N), rng.uniform(1.4, 1.8, N)
    UH = hopf_speed_fast(k, c, tau, xr)
    sigma = rng.uniform(*NOISE["sigma"], N)
    sigma_f = rng.uniform(*NOISE["sigma_f"], N)
    sigma_obs = rng.uniform(*NOISE["sigma_obs"], N)
    lo = 60 if pad else L + 20
    iH = np.where(is_h, rng.integers(lo, n_pts - 100, N), n_pts)
    U0 = rng.uniform(0.35, 0.6, N) * UH
    Uend = U0 + (UH - U0) * (n_pts - 1) / np.maximum(iH, 1)
    Uend = np.where(is_h, Uend, U0)                      # null: constant speed
    U_paths = U0[:, None] + (Uend - U0)[:, None] * np.linspace(0, 1, n_pts)[None, :]
    rec = simulate_batch(U_paths, k, c, tau, xr, sigma, sigma_f, sigma_obs, dt, sub, rng)
    return rec, iH

# Class encoding of DL_ChronosClassifier.py: 0 Fold, 1 Hopf, 2 Transcritical, 3 Null
LABELS = {"hopf": 1, "null": 3}

class PorpoisingData:
    """Same attribute names the classifier script reads from `DL`."""
    pass

def get_porpoising(n_series=400, L=500, seed=0, pad=False):
    """n_series TOTAL (half Hopf, half Null), split 80/10/10 stratified by class.
    Arrays are (N, L, 1) float32, matching DL.train / DL.validation / DL.test."""
    rng = np.random.default_rng(seed)
    per = n_series // len(LABELS)
    X, y = [], []
    for name, lab in LABELS.items():
        for _ in range(per):
            x, iH = make_sample(rng, name, L=L, pad=pad)
            X.append(to_window(x, iH, rng, L=L, pad=pad))
            y.append(lab)
    X = np.array(X, dtype=np.float32)[..., None]
    y = np.array(y, dtype=np.int64)
    P = PorpoisingData()
    idx = {"train": [], "validation": [], "test": []}
    for lab in LABELS.values():
        ii = rng.permutation(np.where(y == lab)[0])
        a, b = int(0.8 * len(ii)), int(0.9 * len(ii))
        idx["train"] += list(ii[:a]); idx["validation"] += list(ii[a:b]); idx["test"] += list(ii[b:])
    for split, ii in idx.items():
        ii = rng.permutation(ii)
        setattr(P, split, X[ii])
        setattr(P, f"{split}_target", y[ii])
    P.test_source = ["porpoising"] * len(P.test)
    return P

def merge_into(DL, P):
    """Append porpoising series to an existing DL object (train/validation/test)."""
    for split in ("train", "validation", "test"):
        setattr(DL, split, np.concatenate([getattr(DL, split), getattr(P, split)]))
        tname = f"{split}_target"
        setattr(DL, tname, np.concatenate([np.asarray(getattr(DL, tname)), getattr(P, tname)]))
    DL.test_source = list(DL.test_source) + P.test_source
    return DL

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400, help="total series (half Hopf, half Null)")
    ap.add_argument("--L", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pad", action="store_true", help="allow left zero-padded short windows")
    ap.add_argument("--out", default="porpoising")
    a = ap.parse_args()
    P = get_porpoising(a.n, a.L, a.seed, a.pad)
    for sp in ("train", "validation", "test"):
        np.save(f"{a.out}_{sp}.npy", getattr(P, sp))
        np.save(f"{a.out}_{sp}_target.npy", getattr(P, f"{sp}_target"))
        print(sp, getattr(P, sp).shape, np.bincount(getattr(P, f"{sp}_target"), minlength=4))