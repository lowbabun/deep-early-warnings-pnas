"""
Shared data loader for dl_newtraining.py and Chronos_n_MLP.py.

Three data sources can be used, independently for the TRAIN(+validation) side
and for the TEST side:

    "library"     the existing simulated library loaded by DL_training.py
    "may"         residual series simulated from May's harvesting model ramped
                  through its upper fold (see may_ews_simulation.py). Every
                  series here is a Fold example (label FOLD_LABEL).
    "porpoising"  residual series from a toy F1 porpoising (ground-effect
                  bouncing) model, see porpoising_ews.py. Half are Hopf examples
                  (speed ramped through the Hopf point, label 1) and half are
                  Null examples (constant speed far below it, label 3).
    "both"        library + may (kept for backward compatibility)
    "all"         library + may + porpoising

Any "+"-joined combination also works, e.g. "library+porpoising".

The returned object has the same attribute names the training scripts already
use from DL_training (train, validation, test, *_target, ts_len), so the only
change needed in a script is:

    DL = get_data(train_data=..., test_data=...)

It also carries train_source / validation_source / test_source arrays
("library", "may" or "porpoising" per series) so plots can show where each
example came from.

Notes
  * May and porpoising series are processed like the library ones: the last
    ts_len points before the tipping point are used (left zero-padded if the
    pre-tipping record is shorter), then divided by the mean |x| of the
    non-zero entries. Porpoising Null series have no tipping point, so the
    last ts_len points of the record are used.
  * Because every May series is Fold, "may" alone is a single-class dataset:
    fine as an out-of-distribution TEST set, not useful as the only TRAIN set.
    "porpoising" alone has two classes (Hopf, Null).
  * May and porpoising series are split 80/10/10 into train/val/test pools
    (porpoising stratified by class); each is an independent noise
    realisation, so no series appears in more than one split.
  * DL_training (and its slow data load / TensorFlow import) is only imported
    when a source that needs it is requested.
"""

import types

import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess

TS_LEN = 500
FOLD_LABEL = 0  # matches class_names = {0: "Fold", 1: "Hopf", 2: "Transcritical", 3: "Null"}
PORP_LABELS = {"hopf": 1, "null": 3}
VALID_SOURCES = ("library", "may", "porpoising")
_ALIASES = {"both": ("library", "may"), "all": ("library", "may", "porpoising")}
VALID_OPTIONS = VALID_SOURCES + tuple(_ALIASES) + ("e.g. 'library+porpoising'",)


def _sources(opt):
    """Turn an option string into a tuple of source names (raises on bad input)."""
    if opt in _ALIASES:
        return _ALIASES[opt]
    parts = tuple(opt.split("+"))
    if not parts or any(p not in VALID_SOURCES for p in parts):
        raise ValueError(f"data option must be one of {VALID_OPTIONS}, got {opt!r}")
    return parts


# ── May's harvesting model: dV/dt = r V (1 - V/K) - 0.1 h V^2 / (V0^2 + V^2) + sigma xi(t) ──
_R, _K, _V0 = 0.333, 25.0, 3.0
_HSTART, _HEND, _TMAX = 15.0, 24.0, 1000.0
_DT, _DT_OUT = 0.01, 1.0
_LOWESS_FRAC = 0.2


def _f(V, h):
    return _R * V * (1 - V / _K) - 0.1 * h * V**2 / (_V0**2 + V**2)


def _h_eq(V):
    """Harvesting rate at which V is an equilibrium."""
    return 10 * _R * (1 - V / _K) * (V**2 + _V0**2) / V


def _upper_fold():
    """(h, V) of the upper fold - the one the ramp reaches first."""
    V = np.linspace(0.01, _K - 0.01, 400001)
    hv = _h_eq(V)
    d = np.diff(hv)
    idx = np.where(np.sign(d[:-1]) != np.sign(d[1:]))[0] + 1
    return max((hv[i], V[i]) for i in idx)


def _upper_equilibrium(h):
    V = _K
    for _ in range(200000):
        V += _f(V, h) * 0.01
    return V


def simulate_may_residuals(n_series, seed=0, sigma=0.3, ts_len=TS_LEN):
    """
    Simulate n_series independent noise realisations of the ramped May model
    (Euler-Maruyama, vectorised across realisations) and return their lowess
    residuals up to the tipping point as an array of shape (n_series, ts_len, 1),
    BEFORE normalisation.
    """
    rng = np.random.default_rng(seed)
    h_fold, _ = _upper_fold()
    t_bif = _TMAX * (h_fold - _HSTART) / (_HEND - _HSTART)

    stride = int(round(_DT_OUT / _DT))
    n_out = int(t_bif // _DT_OUT) + 1                # samples with t <= t_bif
    n_steps = (n_out - 1) * stride + 1

    V = np.full(n_series, _upper_equilibrium(_HSTART))
    out = np.empty((n_out, n_series))
    out[0] = V
    sqrt_dt = np.sqrt(_DT)
    chunk = 4000
    i = 0
    while i < n_steps - 1:
        m = min(chunk, n_steps - 1 - i)
        noise = rng.standard_normal((m, n_series)) * sigma * sqrt_dt
        for k in range(m):
            h = _HSTART + (_HEND - _HSTART) * (i + k) * _DT / _TMAX
            V = np.maximum(V + _f(V, h) * _DT + noise[k], 0.0)
            if (i + k + 1) % stride == 0:
                out[(i + k + 1) // stride] = V
        i += m

    tp = np.arange(n_out) * _DT_OUT
    series = np.zeros((n_series, ts_len, 1))
    for s in range(n_series):
        trend = lowess(out[:, s], tp, frac=_LOWESS_FRAC, return_sorted=False)
        resid = out[:, s] - trend
        if len(resid) >= ts_len:
            series[s, :, 0] = resid[-ts_len:]        # window ending at the tipping point
        else:
            series[s, ts_len - len(resid):, 0] = resid  # left zero-pad
    return series


def _normalise(series):
    """Same normalisation as DL_training: divide by mean |x| over non-zero entries."""
    series = series.copy()
    for i in range(len(series)):
        x = series[i, :, 0]
        nz = x != 0
        if nz.any():
            x[nz] = x[nz] / np.abs(x[nz]).mean()
    return series


def load_may_dataset(n_series=400, seed=0, sigma=0.3, ts_len=TS_LEN):
    """Return {'train': (X, y), 'validation': (X, y), 'test': (X, y)} (80/10/10)."""
    X = _normalise(simulate_may_residuals(n_series, seed=seed, sigma=sigma, ts_len=ts_len))
    y = np.full(n_series, FOLD_LABEL, dtype=int)
    n_train, n_val = int(0.8 * n_series), int(0.1 * n_series)
    return {
        "train": (X[:n_train], y[:n_train]),
        "validation": (X[n_train:n_train + n_val], y[n_train:n_train + n_val]),
        "test": (X[n_train + n_val:], y[n_train + n_val:]),
    }


# ── Porpoising (F1 ground-effect bouncing) toy model: Hopf + Null ────────────
_PORP_VERSION = "v2-batch"     # bump if the generator changes, to invalidate cached files
_PORP_CHUNK = 2000             # series simulated together (bounds memory)


def simulate_porpoising_residuals(n_per_class, seed=0, ts_len=TS_LEN, pad=False, lowess_delta=0.0):
    """
    Returns (X, y): X has shape (2 * n_per_class, ts_len, 1) of lowess residuals
    BEFORE normalisation, y the labels from PORP_LABELS (Hopf=1, Null=3).
    Hopf: the record ends at the Hopf point and the last ts_len points are used.
    Null: the last ts_len points of a constant-speed record are used.
    pad=True allows shorter pre-transition records that are left zero-padded.
    lowess_delta > 0 speeds up the lowess step (in time units; the recording interval
    is 0.5, so delta=2 skips ~4 points between exact fits); 0 = exact, like May's series.
    """
    import porpoising_ews as pw                      # lazy: only needed for this source
    rng = np.random.default_rng(seed)
    names = ["hopf"] * n_per_class + ["null"] * n_per_class
    y = np.array([PORP_LABELS[nm] for nm in names], dtype=int)
    X = np.zeros((len(names), ts_len, 1))
    for start in range(0, len(names), _PORP_CHUNK):
        chunk = names[start:start + _PORP_CHUNK]
        rec, iH = pw.make_batch(rng, chunk, L=ts_len, pad=pad)
        for j in range(len(chunk)):
            r = rec[j, :iH[j]]                       # pre-transition record (all of it for Null)
            tp = np.arange(len(r)) * 0.5             # recording interval of the toy model
            resid = r - lowess(r, tp, frac=_LOWESS_FRAC, delta=lowess_delta, return_sorted=False)
            if len(resid) >= ts_len:
                X[start + j, :, 0] = resid[-ts_len:]
            else:
                X[start + j, ts_len - len(resid):, 0] = resid
        print(f"    porpoising: {min(start + _PORP_CHUNK, len(names))}/{len(names)} series simulated")
    return X, y


def _porp_cache_file(n_series, seed, ts_len, pad, lowess_delta):
    import hashlib, json, os
    import porpoising_ews as pw
    key = json.dumps([_PORP_VERSION, n_series, seed, ts_len, pad, lowess_delta,
                      {k: list(v) for k, v in sorted(pw.NOISE.items())}], sort_keys=True)
    h = hashlib.md5(key.encode()).hexdigest()[:12]
    folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "porpoising_cache")
    return os.path.join(folder, f"porp_n{n_series}_s{seed}_{h}.npz")


def load_porpoising_dataset(n_series=400, seed=0, ts_len=TS_LEN, pad=False,
                            lowess_delta=0.0, cache=True, refresh=False):
    """80/10/10 split, stratified by class (Hopf / Null), half of n_series each.
    With cache=True the normalised series are stored in porpoising_cache/ (keyed on
    n_series, seed, ts_len, pad, lowess_delta and the NOISE ranges) and reloaded next time;
    refresh=True regenerates and overwrites."""
    import os
    path = _porp_cache_file(n_series, seed, ts_len, pad, lowess_delta) if cache else None
    if path is not None and os.path.exists(path) and not refresh:
        d = np.load(path)
        X, y = d["X"], d["y"]
        print(f"  loaded cached porpoising series from {path}")
    else:
        X, y = simulate_porpoising_residuals(n_series // len(PORP_LABELS), seed, ts_len, pad, lowess_delta)
        X = _normalise(X)
        if path is not None:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            np.savez_compressed(path, X=X, y=y)
            print(f"  cached porpoising series to {path}")
    rng = np.random.default_rng(seed + 1)
    idx = {"train": [], "validation": [], "test": []}
    for lab in PORP_LABELS.values():
        ii = rng.permutation(np.where(y == lab)[0])
        a, b = int(0.8 * len(ii)), int(0.9 * len(ii))
        idx["train"] += list(ii[:a]); idx["validation"] += list(ii[a:b]); idx["test"] += list(ii[b:])
    out = {}
    for split, v in idx.items():
        v = rng.permutation(v)
        out[split] = (X[v], y[v])
    return out


def get_data(train_data="library", test_data="library", n_may=400, may_seed=0, may_sigma=0.3,
             n_porp=400, porp_seed=0, porp_pad=False,
             porp_cache=True, porp_refresh=False, porp_lowess_delta=0.0):
    """
    train_data: source for train + validation ("library" | "may" | "porpoising" | "both" | "all" | "a+b")
    test_data:  source for test              (same options)
    n_may:      number of May series to simulate in total (split 80/10/10)
    n_porp:     number of porpoising series in total (half Hopf, half Null; split 80/10/10)
    porp_pad:   allow zero-padded short pre-transition windows for porpoising series
    porp_cache: store/reload the generated porpoising series in porpoising_cache/
    porp_refresh: ignore any cached file and regenerate
    porp_lowess_delta: >0 speeds up the lowess step (approximate); 0 = exact
    """
    train_src, test_src = _sources(train_data), _sources(test_data)
    if train_src == ("may",):
        print("WARNING: training on 'may' alone is a single-class (Fold-only) dataset.")

    used = set(train_src) | set(test_src)

    lib = None
    if "library" in used:
        import DL_training as lib  # slow: loads the full library at import time
    ts_len = lib.ts_len if lib is not None else TS_LEN

    may = None
    if "may" in used:
        print(f"Simulating {n_may} May-model series (sigma={may_sigma}, seed={may_seed})...")
        may = load_may_dataset(n_series=n_may, seed=may_seed, sigma=may_sigma, ts_len=ts_len)
    porp = None
    if "porpoising" in used:
        print(f"Simulating {n_porp} porpoising series (Hopf + Null, seed={porp_seed})...")
        porp = load_porpoising_dataset(n_series=n_porp, seed=porp_seed, ts_len=ts_len, pad=porp_pad,
                                       lowess_delta=porp_lowess_delta, cache=porp_cache, refresh=porp_refresh)

    out = types.SimpleNamespace(ts_len=ts_len)

    def build(split, sources):
        xs, ys, src = [], [], []
        if "library" in sources:
            x = np.asarray(getattr(lib, split))
            y = np.asarray(getattr(lib, split + "_target")).astype(int)
            xs.append(x); ys.append(y); src += ["library"] * len(x)
        if "may" in sources:
            x, y = may[split]
            xs.append(x); ys.append(y); src += ["may"] * len(x)
        if "porpoising" in sources:
            x, y = porp[split]
            xs.append(x); ys.append(y); src += ["porpoising"] * len(x)
        return np.concatenate(xs), np.concatenate(ys), np.array(src)

    for split, option, sources in (("train", train_data, train_src),
                                   ("validation", train_data, train_src),
                                   ("test", test_data, test_src)):
        x, y, src = build(split, sources)
        setattr(out, split, x)
        setattr(out, split + "_target", y)
        setattr(out, split + "_source", src)
        print(f"  {split:<10} [{option}]: {len(x)} series, "
              f"class counts {np.bincount(y, minlength=4).tolist()}")
    return out