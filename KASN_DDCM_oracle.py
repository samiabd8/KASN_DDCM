
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import socket
import tempfile
import gc
import math
import os
import sys
import time
import uuid
from typing import Callable, Sequence

import numpy as np
import pandas as pd
import scipy.stats
from scipy.interpolate import CubicSpline
from scipy.stats import norm as _norm
from scipy.special import ndtr as _ndtr

_INV_SQRT_2PI = 0.3989422804014327

_trapz = getattr(np, "trapezoid", None) or np.trapz


def _npdf(z):
    return _INV_SQRT_2PI * np.exp(-0.5 * z * z)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import TensorDataset, DataLoader
    from sklearn.preprocessing import StandardScaler
    TORCH_AVAILABLE = True
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
except ImportError:
    TORCH_AVAILABLE = False
    DEVICE = None


DGP = 'ev_iid'
_VALID_DGPS = {'ev_iid', 'ar1_gauss'}
assert DGP in _VALID_DGPS, f"unknown DGP {DGP!r}; expected {sorted(_VALID_DGPS)}"

M_STATES   = 90
XI_N_NODES = 7
SIGMA_ETA  = 1.0
SIGMA_NU   = 0.5

TRUTH_EV_IID = dict(alpha1=-0.03, alpha2=-5.0, beta=0.95,
                    eta=(0.36, 0.62, 0.02))
TRUTH_AR1    = dict(alpha1=-0.05, alpha2=-4.0, beta=0.95, rho=0.60,
                    eta=(0.36, 0.62, 0.02))

PRIOR_ALPHA1   = (-0.10, 0.0)
PRIOR_ALPHA2   = (-15.0, -2.0)
PRIOR_BETA     = (0.80, 0.98)
PRIOR_RHO      = (0.0, 0.90)
PRIOR_ETA_CONC = (34.0, 64.0, 2.0)

DESIGN_BETA_A = 3.0
DESIGN_BETA_B = 3.0

DESIGN_BETA_SHAPES = {}

DESIGN_INFLATE = 0.20
DESIGN_INFLATE_BY_COORD = {}

DESIGN_INFLATE_COORDS = ('alpha1', 'alpha2', 'beta', 'rho')
DESIGN_RHO_LO_CAP  = 0.0
DESIGN_ALPHA2_HI_CAP = -0.5
DESIGN_UNIFORM_PARAMS = False

DECOUPLE_POSTERIOR_DESIGN = True
POSTERIOR_DESIGN_KIND     = 'uniform'
MATCH_PRIOR_TO_INFER_BOX  = True
INFER_BOX_FRAC = {'alpha1': 0.30, 'alpha2': 0.20, 'beta': 0.35, 'rho': 0.45}
INFER_ALPHA2_HI_CAP = DESIGN_ALPHA2_HI_CAP
INFER_BETA_HI_CAP   = PRIOR_BETA[1]
INFER_RHO_LO_CAP    = DESIGN_RHO_LO_CAP
INFER_RHO_HI_CAP    = PRIOR_RHO[1]

KASN_SPLINE_ORDER = 4
KASN_GRID_RANGE   = [-0.5, 1.5]
KASN_ZETA_DELTA   = 0.4
KASN_DUAL_REG     = True
KASN_ACTIVE_EDGE_THRESHOLD = 1e-4
KASN_WEIGHT_DECAY = 1e-19
KASN_L1_REG_SCALE = 0.0
KASN_M_SMOOTH     = 1
KASN_C_S_N        = 100.0

ADAPTIVE_GRID          = False
GLOBAL_BEST_RESTORE    = True
KASN_GRID_PHASE_FRACS  = (0.1, 0.2)
KASN_GRID_PHASE_SCALES = (0.25, 0.5)
OSLO_N_EVAL            = 2000

POST_TRAINING_PRUNING        = True
PRUNING_THRESHOLD_METHOD     = 'delta_normalized'
PRUNING_RELATIVE_FRACTION    = 0.01
PRUNE_AWARE_SELECTION        = True
POST_PRUNE_FINETUNE_EPOCHS   = 30
POST_PRUNE_FINETUNE_LR_SCALE = 0.5
POST_PRUNE_FINETUNE_PATIENCE = 15

USE_AMP              = True
INFERENCE_CHUNK_SIZE = 8192
MONITOR_EVAL_R2      = True
MONITOR_IN_HP_SEARCH = False
EPOCH_LOG_EVERY      = 10
MCMC_VERBOSE         = False
MCMC_REPORT_EVERY    = 10

KASN_GAMMA_GRID = (0.30, 0.40)
KASN_LR_GRID    = (5e-4, 1e-3)
KASN_GL_GRID    = (1e-3, 1e-2)
KASN_WIDTH_GRID = (24, 32)

COMPUTE_AVERAGE_DERIVATIVES = True

COMPUTE_STATE_DERIVATIVES = True
COMPUTE_PARAM_DERIVATIVES = True
AD_STATE_LABELS_BY_DGP = {'ev_iid': ['x'], 'ar1_gauss': ['x', 'xi']}
AD_PARAM_LABELS_BY_DGP = {'ev_iid': ['alpha2', 'beta'],
                          'ar1_gauss': ['alpha2', 'beta', 'rho']}

COMPUTE_COUNTERFACTUAL = True
CF_PARAM  = 'alpha2'
CF_TAU    = 0.20
CF_CENTER = None
CF_OMEGA_CLIP = 50.0

AD_RIESZ_METHOD      = 'analytic'
AD_RIESZ_CAP_MULT    = 6.0
AD_RIESZ_WINSOR      = None
AD_RIESZ_N_COS       = 16
AD_RIESZ_RIDGE       = 1e-6
AD_RIESZ_RIDGE_SCALE = 1e-4
AD_RIESZ_CORRECTION  = True
AD_GRAD_CHUNK_SIZE   = 4096
AD_MIN_DISTINCT_FRAC = 0.01
AD_NW_BANDWIDTH      = None

TRAIN_NOISE_SD = 0

OUTPUT_DIR  = "ddcm_mc_results"
SEED        = 2001
SAVE_CHAINS = False
CHAIN_THIN  = 10
MAKE_PLOTS  = False

MODE_MC        = 'standard'
N_TRAIN_OVERRIDE = 10000
N_VAL_OVERRIDE   = 2000
N_TEST_OVERRIDE  = 2000
SELECT_HP      = False
MC_VERBOSE     = True
THREADS        = 1

COMPUTE_TRUTH   = True
N_TRUTH         = 100000
TRUTH_SEED      = 1996
TRUTH_FD_REL    = 1e-3
USE_TRUTH_CACHE = True

COMPUTE_ORACLE_FUNCTIONALS = True

RUN_FFANN        = True
FFANN_HIDDEN     = 100
FFANN_EPOCHS     = 600
FFANN_PATIENCE   = 50
FFANN_LR         = 1e-3
FFANN_BATCH      = 512
FFANN_C_WEIGHT   = 25.0
FFANN_C_OUT      = 25.0
FFANN_MCMC       = True

COMPUTE_PATH_DIAGNOSTICS = True
PATH_N_DRAWS             = 200

RUN_ID = str(uuid.uuid4())[:8]
EULER_MASCHERONI = 0.5772156649015329
KASN_BASE_ACTIVATION = (nn.SiLU if TORCH_AVAILABLE else None)


def _out_path(stem: str, ext: str) -> str:
    return os.path.join(OUTPUT_DIR, f"{stem}_{DGP}_seed{SEED}_{RUN_ID}.{ext}")



class BusEngineEVIID:

    name = 'ev_iid'
    feature_names = ['x', 'alpha1', 'alpha2', 'beta', 'eta1', 'eta2']
    param_names = ['alpha1', 'alpha2', 'beta', 'eta1', 'eta2', 'eta3']

    def __init__(self, M: int = M_STATES) -> None:
        if M < 5:
            raise ValueError("M must be at least 5.")
        self.M = int(M)
        self.input_dim = len(self.feature_names)

    def solve_ev(self, alpha1, alpha2, beta, eta, tol=1e-10, max_iter=10_000):
        eta = np.asarray(eta, dtype=np.float64)
        if eta.shape != (3,):
            raise ValueError(f"eta must have shape (3,), got {eta.shape}.")
        if not np.isclose(eta.sum(), 1.0, atol=1e-6):
            raise ValueError(f"eta must sum to 1, got {eta.sum():.6f}")
        if not (0.0 <= beta < 1.0):
            raise ValueError(f"beta must lie in [0, 1), got {beta}")
        EV = np.zeros(self.M); EV_new = np.empty_like(EV)
        for it in range(max_iter):
            self._bellman(EV, EV_new, alpha1, alpha2, beta, eta)
            err = np.max(np.abs(EV_new - EV)); EV[:] = EV_new
            if err < tol:
                return EV, it + 1
        return EV, max_iter

    def _shift(self, EV):
        p1 = np.empty_like(EV); p1[:-1] = EV[1:]; p1[-1] = EV[-1]
        p2 = np.empty_like(EV); p2[:-2] = EV[2:]; p2[-2:] = EV[-1]
        return p1, p2

    def _bellman(self, EV, EV_new, alpha1, alpha2, beta, eta):
        p1, p2 = self._shift(EV)
        ev1 = eta[0] * EV + eta[1] * p1 + eta[2] * p2
        ev2 = eta[0] * EV[0] + eta[1] * EV[1] + eta[2] * EV[2]
        x = np.arange(1, self.M + 1, dtype=np.float64)
        v1 = alpha1 * x + beta * ev1
        v2 = alpha2 + beta * ev2
        v1[-1] = -np.inf
        vmax = np.maximum(v1, v2)
        EV_new[:] = (vmax + np.log(np.exp(v1 - vmax) + np.exp(v2 - vmax))
                     + EULER_MASCHERONI)

    def compute_F(self, EV, beta, eta):
        eta = np.asarray(eta, dtype=np.float64)
        p1, p2 = self._shift(EV)
        ev = eta[0] * EV + eta[1] * p1 + eta[2] * p2
        ev2 = beta * (eta[0] * EV[0] + eta[1] * EV[1] + eta[2] * EV[2])
        return beta * ev - ev2

    def F_grid(self, q):
        EV, _ = self.solve_ev(q['alpha1'], q['alpha2'], q['beta'], q['eta'])
        return self.compute_F(EV, q['beta'], q['eta'])

    def F_at(self, X, eta_full):
        xg = np.arange(1, self.M + 1, dtype=np.float64)
        out = np.empty(X.shape[0])
        for i in range(X.shape[0]):
            Fg = self.F_grid(dict(alpha1=X[i, 1], alpha2=X[i, 2],
                                  beta=X[i, 3], eta=eta_full[i]))
            out[i] = float(CubicSpline(xg, Fg, bc_type='natural')(X[i, 0]))
        return out

    def ccp_replace(self, q):
        EV, _ = self.solve_ev(q['alpha1'], q['alpha2'], q['beta'], q['eta'])
        eta = np.asarray(q['eta'], dtype=np.float64)
        p1, p2 = self._shift(EV)
        ev1 = eta[0] * EV + eta[1] * p1 + eta[2] * p2
        ev2 = eta[0] * EV[0] + eta[1] * EV[1] + eta[2] * EV[2]
        x = np.arange(1, self.M + 1, dtype=np.float64)
        v1 = q['alpha1'] * x + q['beta'] * ev1
        v2 = q['alpha2'] + q['beta'] * ev2
        v1[-1] = -np.inf
        vmax = np.maximum(v1, v2)
        e1, e2 = np.exp(v1 - vmax), np.exp(v2 - vmax)
        return e2 / (e1 + e2)

    def simulate_panel(self, q, n_buses, n_periods, rng):
        eta = np.asarray(q['eta'], dtype=np.float64)
        P2 = self.ccp_replace(q)
        x = np.empty((n_periods, n_buses), dtype=np.int64)
        d = np.empty((n_periods, n_buses), dtype=np.int64)
        x[0] = 1
        for t in range(n_periods):
            d[t] = np.where(rng.uniform(size=n_buses) < P2[x[t] - 1], 2, 1)
            if t < n_periods - 1:
                jumps = rng.choice(3, p=eta, size=n_buses)
                post = np.where(d[t] == 1, x[t], 1)
                x[t + 1] = np.minimum(post + jumps, self.M)
        return {"x": x, "d": d}



def _rouwenhorst(n: int, rho: float, sigma_nu: float):
    if n < 2:
        raise ValueError("need at least two nodes")
    rho = float(np.clip(rho, 0.0, 0.999))
    p = (1.0 + rho) / 2.0
    P = np.array([[p, 1 - p], [1 - p, p]])
    for k in range(3, n + 1):
        Pn = np.zeros((k, k))
        Pn[:-1, :-1] += p * P
        Pn[:-1, 1:] += (1 - p) * P
        Pn[1:, :-1] += (1 - p) * P
        Pn[1:, 1:] += p * P
        Pn[1:-1, :] /= 2.0
        P = Pn
    sigma_stat = sigma_nu / np.sqrt(max(1.0 - rho ** 2, 1e-12))
    psi = sigma_stat * np.sqrt(n - 1.0)
    return np.linspace(-psi, psi, n), P


class BusEngineAR1Gauss:

    name = 'ar1_gauss'
    feature_names = ['x', 'xi', 'alpha1', 'alpha2', 'beta', 'rho',
                     'eta1', 'eta2']
    param_names = ['alpha1', 'alpha2', 'beta', 'rho', 'eta1', 'eta2', 'eta3']

    def __init__(self, M=M_STATES, K=XI_N_NODES, sigma_eta=SIGMA_ETA,
                 sigma_nu=SIGMA_NU):
        self.M, self.K = int(M), int(K)
        self.sigma_eta, self.sigma_nu = float(sigma_eta), float(sigma_nu)
        self.input_dim = len(self.feature_names)
        self._W_cache = None
        self._jidx = None
        self.warm_start = True

    def xi_chain(self, rho):
        return _rouwenhorst(self.K, rho, self.sigma_nu)

    def xi_support(self, rho):
        return float(self.xi_chain(rho)[0][-1])

    def _jump_index(self):
        if getattr(self, '_jidx', None) is None:
            a = np.arange(self.M)
            self._jidx = [np.minimum(a + j, self.M - 1) for j in range(3)]
        return self._jidx

    def _continuations(self, W, P, eta):
        EW = W @ P.T
        idx = self._jump_index()
        C1 = (eta[0] * EW[idx[0], :] + eta[1] * EW[idx[1], :]
              + eta[2] * EW[idx[2], :])
        C2 = eta[0] * EW[0, :] + eta[1] * EW[1, :] + eta[2] * EW[2, :]
        return C1, C2

    def solve_W(self, q, tol=1e-9, max_iter=5_000, W_init=None):
        eta = np.asarray(q['eta'], dtype=np.float64)
        beta, s = float(q['beta']), self.sigma_eta
        grid, P = self.xi_chain(q['rho'])
        x = np.arange(1, self.M + 1, dtype=np.float64)[:, None]
        if W_init is None and self.warm_start and self._W_cache is not None:
            W_init = self._W_cache
        W = (np.zeros((self.M, self.K)) if W_init is None
             else np.array(W_init, dtype=np.float64, copy=True))
        if W.shape != (self.M, self.K):
            W = np.zeros((self.M, self.K))
        for it in range(max_iter):
            C1, C2 = self._continuations(W, P, eta)
            A = q['alpha1'] * x + grid[None, :] + beta * C1
            B = q['alpha2'] + beta * C2[None, :]
            m = B - A; z = m / s
            Wn = A + m * _ndtr(z) + s * _npdf(z)
            Wn[self.M - 1, :] = B[0, :]
            err = float(np.max(np.abs(Wn - W))); W = Wn
            if err < tol:
                if self.warm_start:
                    self._W_cache = W
                return W, it + 1
        if self.warm_start:
            self._W_cache = W
        return W, max_iter

    def F_grid(self, q):
        eta = np.asarray(q['eta'], dtype=np.float64)
        _, P = self.xi_chain(q['rho'])
        W, _ = self.solve_W(q)
        C1, C2 = self._continuations(W, P, eta)
        return float(q['beta']) * (C1 - C2[None, :])

    def F_at(self, X, eta_full):
        xg = np.arange(1, self.M + 1, dtype=np.float64)
        out = np.empty(X.shape[0])
        for i in range(X.shape[0]):
            q = dict(alpha1=X[i, 2], alpha2=X[i, 3], beta=X[i, 4],
                     rho=X[i, 5], eta=eta_full[i])
            Fg = self.F_grid(q)
            grid, _ = self.xi_chain(q['rho'])
            col = CubicSpline(grid, Fg, axis=1,
                              bc_type='natural')(float(X[i, 1]))
            out[i] = float(CubicSpline(xg, col, bc_type='natural')(X[i, 0]))
        return out

    def ccp_replace_grid(self, q, F=None):
        grid, _ = self.xi_chain(q['rho'])
        F = self.F_grid(q) if F is None else F
        x = np.arange(1, self.M + 1, dtype=np.float64)[:, None]
        diff = q['alpha1'] * x + grid[None, :] - q['alpha2'] + F
        P2 = 1.0 - _ndtr(diff / self.sigma_eta)
        P2[self.M - 1, :] = 1.0
        return np.clip(P2, 1e-12, 1 - 1e-12)

    def simulate_panel(self, q, n_buses, n_periods, rng):
        eta = np.asarray(q['eta'], dtype=np.float64)
        _, P = self.xi_chain(q['rho'])
        P2 = self.ccp_replace_grid(q)
        cum = np.cumsum(P, axis=1)
        x = np.empty((n_periods, n_buses), dtype=np.int64)
        d = np.empty((n_periods, n_buses), dtype=np.int64)
        k = rng.integers(0, self.K, size=n_buses)
        x[0] = 1
        for t in range(n_periods):
            d[t] = np.where(rng.uniform(size=n_buses) < P2[x[t] - 1, k], 2, 1)
            if t < n_periods - 1:
                jumps = rng.choice(3, p=eta, size=n_buses)
                post = np.where(d[t] == 1, x[t], 1)
                x[t + 1] = np.minimum(post + jumps, self.M)
                u = rng.uniform(size=n_buses)
                k = np.clip([int(np.searchsorted(cum[k[b]], u[b]))
                             for b in range(n_buses)], 0, self.K - 1)
        return {"x": x, "d": d}



class DesignDistribution:

    def __init__(self, dgp, a=DESIGN_BETA_A, b=DESIGN_BETA_B):
        self.dgp, self.a, self.b = dgp, float(a), float(b)
        if not (self.a > 1.0 and self.b > 1.0):
            raise ValueError("Beta shapes must exceed one for condition (E.1).")
        self.names = list(dgp.feature_names)
        self.shapes = {n: tuple(float(v) for v in
                                DESIGN_BETA_SHAPES.get(n, (self.a, self.b)))
                       for n in self.names}
        self.eta_conc = np.asarray(PRIOR_ETA_CONC, dtype=np.float64)
        kind = 'unif' if DESIGN_UNIFORM_PARAMS else 'beta'
        hw = dgp.xi_support(PRIOR_RHO[1]) if dgp.name == 'ar1_gauss' else 0.0

        def _wide(name, lo, hi, lo_cap=None, hi_cap=None):
            infl = (DESIGN_INFLATE_BY_COORD.get(name, DESIGN_INFLATE)
                    if name in DESIGN_INFLATE_COORDS else 0.0)
            d = infl * (hi - lo)
            lo2, hi2 = lo - d, hi + d
            if lo_cap is not None:
                lo2 = max(lo2, lo_cap)
            if hi_cap is not None:
                hi2 = min(hi2, hi_cap)
            return lo2, hi2

        spec = {'x': (1.0, float(dgp.M), 'beta'),
                'xi': (-hw, hw, 'beta'),
                'alpha1': (*_wide('alpha1', *PRIOR_ALPHA1), kind),
                'alpha2': (*_wide('alpha2', *PRIOR_ALPHA2,
                                  hi_cap=DESIGN_ALPHA2_HI_CAP), kind),
                'beta': (*_wide('beta', *PRIOR_BETA, hi_cap=PRIOR_BETA[1]), kind),
                'rho': (*_wide('rho', *PRIOR_RHO, lo_cap=DESIGN_RHO_LO_CAP,
                               hi_cap=PRIOR_RHO[1]), kind)}
        self.spec = {n: spec[n] for n in self.names if n in spec}
        self.prior_spec = {'x': (1.0, float(dgp.M)), 'xi': (-hw, hw),
                           'alpha1': PRIOR_ALPHA1, 'alpha2': PRIOR_ALPHA2,
                           'beta': PRIOR_BETA, 'rho': PRIOR_RHO}

    def shape(self, coord):
        name = coord if isinstance(coord, str) else self.names[coord]
        return self.shapes.get(name, (self.a, self.b))

    def sample(self, n, rng):
        cols, eta = [], rng.dirichlet(self.eta_conc, size=n)
        for name in self.names:
            if name in ('eta1', 'eta2'):
                cols.append(eta[:, 0 if name == 'eta1' else 1]); continue
            lo, hi, kind = self.spec[name]
            a_n, b_n = self.shape(name)
            u = (rng.uniform(size=n) if kind == 'unif'
                 else rng.beta(a_n, b_n, size=n))
            cols.append(lo + u * (hi - lo))
        return np.column_stack(cols), eta

    def weight_on_ranks(self, r, coord):
        r = np.clip(np.asarray(r, dtype=np.float64), 1e-12, 1 - 1e-12)
        name = self.names[coord]
        if name in ('eta1', 'eta2'):
            a = self.eta_conc[0 if name == 'eta1' else 1]
            b = self.eta_conc.sum() - a
            return scipy.stats.beta.pdf(scipy.stats.beta.ppf(r, a, b), a, b)
        lo, hi, kind = self.spec[name]
        span = hi - lo
        if kind == 'unif':
            return np.full_like(r, 1.0 / span)
        a_n, b_n = self.shape(name)
        u = scipy.stats.beta.ppf(r, a_n, b_n)
        return scipy.stats.beta.pdf(u, a_n, b_n) / span

    def riesz_rms(self, coord):
        if not hasattr(self, '_riesz_rms_cache'):
            self._riesz_rms_cache = {}
        if coord in self._riesz_rms_cache:
            return self._riesz_rms_cache[coord]
        name = self.names[coord]
        if name in ('eta1', 'eta2') or self.spec[name][2] == 'unif':
            val = np.nan
        else:
            lo, hi, _ = self.spec[name]
            a, b = self.shape(name)
            u = np.linspace(1e-9, 1 - 1e-9, 200_001)
            f = scipy.stats.beta.pdf(u, a, b)
            fp = f * ((a - 1.0) / u - (b - 1.0) / (1.0 - u))
            integ = float(_trapz(
                np.where(f > 1e-300, fp ** 2 / np.maximum(f, 1e-300), 0.0), u))
            val = float(np.sqrt(max(integ, 0.0)) / (hi - lo))
        self._riesz_rms_cache[coord] = val
        return val

    def satisfies_E1(self, coord):
        name = self.names[coord]
        if name in ('eta1', 'eta2'):
            a = self.eta_conc[0 if name == 'eta1' else 1]
            return bool(a > 1.0 and (self.eta_conc.sum() - a) > 1.0)
        a_n, b_n = self.shape(name)
        return bool(self.spec[name][2] == 'beta' and a_n > 1.0 and b_n > 1.0)


def ad_labels_for(dgp):
    out = []
    if COMPUTE_STATE_DERIVATIVES:
        out += AD_STATE_LABELS_BY_DGP.get(dgp.name, [])
    if COMPUTE_PARAM_DERIVATIVES:
        out += AD_PARAM_LABELS_BY_DGP.get(dgp.name, [])
    return [l for l in out if l in dgp.feature_names]


def build_training_data(dgp, design, n_train, n_val, n_test, rng,
                        noise_sd=TRAIN_NOISE_SD, verbose=True):
    n_total = n_train + n_val + n_test
    t0 = time.time()
    X, eta_full = design.sample(n_total, rng)
    y = dgp.F_at(X, eta_full)
    if noise_sd > 0.0:
        y = y + noise_sd * rng.standard_normal(n_total)
    assert X.shape[1] == dgp.input_dim, "design/network dimension mismatch"
    assert np.isfinite(y).all(), "non-finite training targets"
    if verbose:
        fails = [design.names[j] for j in range(X.shape[1])
                 if not design.satisfies_E1(j)]
        print(f"        {n_total:,} design points solved in {time.time()-t0:.1f}s"
              f"   d = {X.shape[1]}")
        print(f"        targets: mean={y.mean():+.4f} sd={y.std():.4f}  "
              f"noise_sd={noise_sd}")
        print(f"        (E.1) fails for: {fails if fails else 'no coordinate'}")
    a, b = n_train, n_train + n_val
    return {"X_train": X[:a], "y_train": y[:a], "X_val": X[a:b], "y_val": y[a:b],
            "X_test": X[b:], "y_test": y[b:]}


def infer_boxes(truth_q):
    prior = {'alpha1': PRIOR_ALPHA1, 'alpha2': PRIOR_ALPHA2,
             'beta': PRIOR_BETA, 'rho': PRIOR_RHO}
    lo_cap = {'rho': INFER_RHO_LO_CAP}
    hi_cap = {'alpha2': INFER_ALPHA2_HI_CAP, 'beta': INFER_BETA_HI_CAP,
              'rho': INFER_RHO_HI_CAP}
    out = {}
    for name, (plo, phi) in prior.items():
        if name not in truth_q:
            continue
        frac = INFER_BOX_FRAC.get(name)
        if frac is None:
            out[name] = (plo, phi)
            continue
        c = float(truth_q[name])
        h = frac * (phi - plo)
        lo, hi = max(c - h, plo), min(c + h, phi)
        if name in lo_cap:
            lo = max(lo, lo_cap[name])
        if name in hi_cap:
            hi = min(hi, hi_cap[name])
        if not (lo < hi):
            lo, hi = plo, phi
        out[name] = (lo, hi)
    return out


def build_posterior_training_data(dgp, truth_q, boxes, n_train, n_val, n_test,
                                  rng, kind=POSTERIOR_DESIGN_KIND,
                                  noise_sd=TRAIN_NOISE_SD, verbose=True):
    names = list(dgp.feature_names)
    n_total = n_train + n_val + n_test
    t0 = time.time()

    def draw(lo, hi):
        if kind == 'beta':
            u = rng.beta(DESIGN_BETA_A, DESIGN_BETA_B, size=n_total)
        else:
            u = rng.uniform(size=n_total)
        return lo + u * (hi - lo)

    rho_hi = boxes.get('rho', PRIOR_RHO)[1] if dgp.name == 'ar1_gauss' else 0.0
    hw = dgp.xi_support(rho_hi) if dgp.name == 'ar1_gauss' else 0.0
    eta = rng.dirichlet(np.asarray(PRIOR_ETA_CONC, dtype=np.float64), size=n_total)
    cols = []
    for name in names:
        if name == 'x':
            cols.append(1.0 + rng.uniform(size=n_total) * (dgp.M - 1.0))
        elif name == 'xi':
            cols.append(-hw + rng.uniform(size=n_total) * (2.0 * hw))
        elif name == 'eta1':
            cols.append(eta[:, 0])
        elif name == 'eta2':
            cols.append(eta[:, 1])
        else:
            lo, hi = boxes.get(name, {'alpha1': PRIOR_ALPHA1,
                                      'alpha2': PRIOR_ALPHA2, 'beta': PRIOR_BETA,
                                      'rho': PRIOR_RHO}[name])
            cols.append(draw(lo, hi))
    X = np.column_stack(cols)
    y = dgp.F_at(X, eta)
    if noise_sd > 0.0:
        y = y + noise_sd * rng.standard_normal(n_total)
    assert X.shape[1] == dgp.input_dim, "posterior design/network dim mismatch"
    assert np.isfinite(y).all(), "non-finite posterior training targets"
    if verbose:
        box_str = ", ".join(f"{k}:[{v[0]:.3g},{v[1]:.3g}]"
                            for k, v in boxes.items() if k in names)
        print(f"        posterior design: {n_total:,} pts ({kind}) in "
              f"{time.time()-t0:.1f}s  boxes {box_str}")
        print(f"        targets: mean={y.mean():+.4f} sd={y.std():.4f}")
    a, b = n_train, n_train + n_val
    return {"X_train": X[:a], "y_train": y[:a], "X_val": X[a:b], "y_val": y[a:b],
            "X_test": X[b:], "y_test": y[b:]}



def _softplus(x):
    return np.where(x > 0, x + np.log1p(np.exp(-x)), np.log1p(np.exp(x)))


class MCMCPrior:

    def __init__(self, dgp, boxes=None):
        self.has_rho = (dgp.name == 'ar1_gauss')
        self.eta_conc = np.asarray(PRIOR_ETA_CONC, dtype=np.float64)
        b = boxes or {}
        self.a1 = tuple(b.get('alpha1', PRIOR_ALPHA1))
        self.a2 = tuple(b.get('alpha2', PRIOR_ALPHA2))
        self.bt = tuple(b.get('beta', PRIOR_BETA))
        self.rh = tuple(b.get('rho', PRIOR_RHO))

    def in_support(self, q):
        ok = (self.a1[0] <= q['alpha1'] <= self.a1[1]
              and self.a2[0] <= q['alpha2'] <= self.a2[1]
              and self.bt[0] <= q['beta'] <= self.bt[1]
              and np.all(q['eta'] > 0.0) and abs(q['eta'].sum() - 1.0) < 1e-9)
        if self.has_rho:
            ok = ok and (self.rh[0] <= q['rho'] <= self.rh[1])
        return bool(ok)

    def log_pdf(self, q):
        if not self.in_support(q):
            return -np.inf
        return float(scipy.stats.dirichlet.logpdf(q['eta'], self.eta_conc))

    def sample(self, rng):
        q = dict(alpha1=float(rng.uniform(*self.a1)),
                 alpha2=float(rng.uniform(*self.a2)),
                 beta=float(rng.uniform(*self.bt)),
                 eta=rng.dirichlet(self.eta_conc))
        if self.has_rho:
            q['rho'] = float(rng.uniform(*self.rh))
        return q


class LikelihoodEVIID:

    def __init__(self, panel, M):
        self.x = np.asarray(panel['x'], dtype=np.int64)
        self.d = np.asarray(panel['d'], dtype=np.int64)
        self.M = int(M)
        if ((self.x == M) & (self.d == 1)).any():
            raise ValueError("Inconsistent: d=1 observed at the boundary x=M.")
        post = np.where(self.d == 1, self.x, 1)
        self._pp, self._xn = post[:-1], self.x[1:]
        self._xf, self._df = self.x.ravel(), self.d.ravel()
        self._mask = (self._xf == M)

    def __call__(self, q, F):
        z = q['alpha1'] * self._xf - q['alpha2'] + F[self._xf - 1]
        lp = np.where(self._df == 1, -_softplus(-z), -_softplus(z))
        lp[self._mask] = 0.0
        pt = np.zeros_like(self._xn, dtype=np.float64)
        for j in range(3):
            pt += q['eta'][j] * (np.minimum(self._pp + j, self.M) == self._xn)
        return float(lp.sum()) + float(np.log(np.maximum(pt, 1e-300)).sum())


class LikelihoodAR1:

    def __init__(self, panel, dgp):
        self.x = np.asarray(panel['x'], dtype=np.int64)
        self.d = np.asarray(panel['d'], dtype=np.int64)
        self.dgp = dgp
        self.M, self.K = dgp.M, dgp.K
        self.T, self.N = self.x.shape
        post = np.where(self.d == 1, self.x, 1)
        self._pp, self._xn = post[:-1], self.x[1:]

    def __call__(self, q, F):
        P2 = self.dgp.ccp_replace_grid(q, F=F)
        _, P = self.dgp.xi_chain(q['rho'])
        alpha = np.full((self.N, self.K), 1.0 / self.K)
        logL = 0.0
        for t in range(self.T):
            if t > 0:
                alpha = alpha @ P
            row = P2[self.x[t] - 1, :]
            alpha = alpha * np.where((self.d[t] == 2)[:, None], row, 1.0 - row)
            c = alpha.sum(axis=1)
            if np.any(c <= 0):
                return -np.inf
            logL += float(np.log(c).sum())
            alpha = alpha / c[:, None]
        pt = np.zeros_like(self._xn, dtype=np.float64)
        for j in range(3):
            pt += q['eta'][j] * (np.minimum(self._pp + j, self.M) == self._xn)
        return logL + float(np.log(np.maximum(pt, 1e-300)).sum())


def alr(eta):
    return np.log(eta[:-1] / eta[-1])


def alr_inv(y):
    e = np.exp(y); s = 1.0 + e.sum()
    out = np.empty(3); out[:2] = e / s; out[2] = 1.0 / s
    return out


def log_jacobian_alr(eta):
    return float(np.log(eta).sum())


class DDCMSampler:

    def __init__(self, likelihood, prior, F_oracle, has_rho,
                 sigma_alpha=(8e-3, 1.0), sigma_beta=0.03, sigma_rho=0.05,
                 sigma_eta=0.15, target_acc=0.30, adapt_window=50):
        self.lik, self.prior, self.F_oracle = likelihood, prior, F_oracle
        self.has_rho = has_rho
        self.s_a = np.asarray(sigma_alpha, dtype=np.float64)
        self.s_b, self.s_r, self.s_e = sigma_beta, sigma_rho, sigma_eta
        self.target, self.window = target_acc, adapt_window

    def _post(self, q):
        lp = self.prior.log_pdf(q)
        if not np.isfinite(lp):
            return -np.inf, lp
        Fv = self.F_oracle(q)
        if not np.all(np.isfinite(Fv)):
            return -np.inf, lp
        return self.lik(q, Fv), lp

    def sample(self, n_iter, n_burn, rng=None, verbose=None, label=''):
        rng = rng or np.random.default_rng()
        verbose = MCMC_VERBOSE if verbose is None else verbose
        t_start = time.time()
        for _ in range(300):
            q = self.prior.sample(rng)
            ll, lp = self._post(q)
            if np.isfinite(ll) and np.isfinite(lp):
                break
        else:
            raise RuntimeError("Failed to find an interior starting value.")
        lj = log_jacobian_alr(q['eta'])

        n_tot = n_burn + n_iter
        keys = ['alpha1', 'alpha2', 'beta'] + (['rho'] if self.has_rho else [])
        chain = {k: np.empty(n_tot) for k in keys}
        chain['eta'] = np.empty((n_tot, 3))
        chain['log_post'] = np.empty(n_tot)
        s_a, s_b, s_r, s_e = self.s_a.copy(), self.s_b, self.s_r, self.s_e
        acc = dict(alpha=0, beta=0, rho=0, eta=0)
        win = dict(alpha=0, beta=0, rho=0, eta=0, n=0)

        for it in range(n_tot):
            qp = dict(q)
            qp['alpha1'] = q['alpha1'] + s_a[0] * rng.standard_normal()
            qp['alpha2'] = q['alpha2'] + s_a[1] * rng.standard_normal()
            llp, lpp = self._post(qp)
            if np.log(rng.uniform()) < (llp + lpp) - (ll + lp):
                q, ll, lp = qp, llp, lpp; acc['alpha'] += 1; win['alpha'] += 1

            qp = dict(q); qp['beta'] = q['beta'] + s_b * rng.standard_normal()
            llp, lpp = self._post(qp)
            if np.log(rng.uniform()) < (llp + lpp) - (ll + lp):
                q, ll, lp = qp, llp, lpp; acc['beta'] += 1; win['beta'] += 1

            if self.has_rho:
                qp = dict(q); qp['rho'] = q['rho'] + s_r * rng.standard_normal()
                llp, lpp = self._post(qp)
                if np.log(rng.uniform()) < (llp + lpp) - (ll + lp):
                    q, ll, lp = qp, llp, lpp; acc['rho'] += 1; win['rho'] += 1

            qp = dict(q)
            qp['eta'] = alr_inv(alr(q['eta']) + s_e * rng.standard_normal(2))
            llp, lpp = self._post(qp)
            ljp = log_jacobian_alr(qp['eta'])
            if np.log(rng.uniform()) < (llp + lpp + ljp) - (ll + lp + lj):
                q, ll, lp, lj = qp, llp, lpp, ljp
                acc['eta'] += 1; win['eta'] += 1

            for k in keys:
                chain[k][it] = q[k]
            chain['eta'][it] = q['eta']
            chain['log_post'][it] = ll + lp
            win['n'] += 1

            if it < n_burn and win['n'] >= self.window:
                s_a *= np.exp(0.5 * (win['alpha'] / win['n'] - self.target))
                s_b *= np.exp(0.5 * (win['beta'] / win['n'] - self.target))
                s_r *= np.exp(0.5 * (win['rho'] / win['n'] - self.target))
                s_e *= np.exp(0.5 * (win['eta'] / win['n'] - self.target))
                win = dict(alpha=0, beta=0, rho=0, eta=0, n=0)

            step = max(1, int(n_tot * MCMC_REPORT_EVERY / 100))
            if verbose and (it + 1) % step == 0:
                el = time.time() - t_start
                eta_s = el * (n_tot - it - 1) / max(it + 1, 1)
                den = max(it + 1, 1)
                phase = 'burn' if it < n_burn else 'samp'
                rates = (f"a={acc['alpha']/den:.2f} b={acc['beta']/den:.2f} "
                         + (f"r={acc['rho']/den:.2f} " if self.has_rho else "")
                         + f"e={acc['eta']/den:.2f}")
                extra = (f"  rho={q['rho']:+.3f}" if self.has_rho else "")
                print(f"      {label}[{phase} {it+1:6d}/{n_tot}] "
                      f"logpost={ll+lp:11.1f}  "
                      f"a1={q['alpha1']:+.4f} a2={q['alpha2']:+.2f} "
                      f"b={q['beta']:.3f}{extra}  acc({rates})  "
                      f"{el:5.1f}s elapsed, {eta_s:5.1f}s left")

        out = {k: v[n_burn:] for k, v in chain.items()}
        diag = {f"acc_{k}": acc[k] / n_tot for k in acc}
        diag.update(sigma_alpha=s_a.tolist(), sigma_beta=s_b,
                    sigma_rho=s_r, sigma_eta=s_e)
        return out, diag


def make_F_oracle_exact(dgp):
    return lambda q: dgp.F_grid(q)


def make_F_oracle_kasn(dgp, model):
    if dgp.name == 'ev_iid':
        base = np.empty((dgp.M, dgp.input_dim))
        base[:, 0] = np.arange(1, dgp.M + 1, dtype=np.float64)

        def F_fn(q):
            X = base.copy()
            X[:, 1], X[:, 2], X[:, 3] = q['alpha1'], q['alpha2'], q['beta']
            X[:, 4], X[:, 5] = q['eta'][0], q['eta'][1]
            return model.predict(X)
        return F_fn

    def F_fn_ar1(q):
        grid, _ = dgp.xi_chain(q['rho'])
        xx, kk = np.meshgrid(np.arange(1, dgp.M + 1, dtype=np.float64), grid,
                             indexing='ij')
        X = np.empty((xx.size, dgp.input_dim))
        X[:, 0], X[:, 1] = xx.ravel(), kk.ravel()
        X[:, 2], X[:, 3] = q['alpha1'], q['alpha2']
        X[:, 4], X[:, 5] = q['beta'], q['rho']
        X[:, 6], X[:, 7] = q['eta'][0], q['eta'][1]
        return model.predict(X).reshape(dgp.M, dgp.K)
    return F_fn_ar1



_EVAL_MONITOR = None


def set_eval_monitor(X_val=None, y_val=None, X_test=None, y_test=None,
                     active=True):
    global _EVAL_MONITOR
    if not active or X_val is None:
        _EVAL_MONITOR = None
        return
    _EVAL_MONITOR = {'X_val': np.asarray(X_val),
                     'y_val': np.asarray(y_val).ravel(),
                     'X_test': None if X_test is None else np.asarray(X_test),
                     'y_test': None if y_test is None else np.asarray(y_test).ravel()}
    print("  [Monitor] in-training R2 on val and test enabled "
          "(diagnostic only, not used for selection)")


def clear_eval_monitor():
    global _EVAL_MONITOR
    _EVAL_MONITOR = None


def _r2(y, yhat):
    y = np.asarray(y, float).ravel(); yhat = np.asarray(yhat, float).ravel()
    ss = float(np.sum((y - y.mean()) ** 2))
    return float(1.0 - np.sum((y - yhat) ** 2) / ss) if ss > 0 else np.nan


def _eval_monitor_str(model):
    if _EVAL_MONITOR is None:
        return ''
    m, parts = _EVAL_MONITOR, []
    was_training = model.training
    try:
        model.eval()
        for split in ('val', 'test'):
            X, y = m[f'X_{split}'], m[f'y_{split}']
            if X is None or y is None:
                continue
            pred = model.predict(X)
            if pred.shape[0] == y.shape[0]:
                parts.append(f"{split[0].upper()}:R2={_r2(y, pred):+.4f}")
    except Exception as exc:
        return f'  [monitor unavailable: {type(exc).__name__}]'
    finally:
        if was_training:
            model.train()
    return ('  ' + '  '.join(parts)) if parts else ''



if TORCH_AVAILABLE:

    class EmpiricalCDFTransformer:

        def __init__(self):
            self.sorted_values_, self.n_train_ = None, None

        def fit(self, X):
            X = X.reshape(-1, 1) if X.ndim == 1 else X
            self.sorted_values_, self.n_train_ = np.sort(X, axis=0), X.shape[0]
            return self

        def transform(self, X):
            X = X.reshape(-1, 1) if X.ndim == 1 else X
            n = self.n_train_
            cdf = np.zeros_like(X, dtype=np.float64)
            for i in range(X.shape[1]):
                cdf[:, i] = np.searchsorted(self.sorted_values_[:, i], X[:, i],
                                            side='right') / (n + 1.0)
            return np.clip(cdf, 0.0, 1.0)

        def fit_transform(self, X):
            return self.fit(X).transform(X)

    class BSplineBasis(nn.Module):

        def __init__(self, in_features, grid_size=5,
                     spline_order=KASN_SPLINE_ORDER, grid_range=KASN_GRID_RANGE):
            super().__init__()
            self.in_features, self.grid_size = in_features, grid_size
            self.spline_order = spline_order
            self.register_buffer("grid", self._grid(grid_range, grid_size))

        def _grid(self, gr, gs):
            h = (gr[1] - gr[0]) / gs
            g = torch.arange(-self.spline_order,
                             gs + self.spline_order + 1) * h + gr[0]
            return g.expand(self.in_features, -1).contiguous()

        def b_splines(self, x):
            grid = self.grid
            x = x.unsqueeze(-1)
            bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
            for k in range(1, self.spline_order + 1):
                left = (x - grid[:, :-(k + 1)]) / (
                    grid[:, k:-1] - grid[:, :-(k + 1)]).clamp_min(1e-8)
                right = (grid[:, k + 1:] - x) / (
                    grid[:, k + 1:] - grid[:, 1:(-k)]).clamp_min(1e-8)
                bases = left * bases[:, :, :-1] + right * bases[:, :, 1:]
            return bases.contiguous()

        def forward(self, x):
            s = x.shape
            return self.b_splines(x.reshape(-1, self.in_features)).reshape(*s[:-1], -1)

    def _oslo_projection_matrix(G_old, G_new, k=KASN_SPLINE_ORDER,
                                grid_range=KASN_GRID_RANGE, n_eval=OSLO_N_EVAL):
        if G_old == G_new:
            return torch.eye(G_old + k)
        xe = torch.linspace(grid_range[0] + 1e-4, grid_range[1] - 1e-4,
                            n_eval).unsqueeze(-1)
        with torch.no_grad():
            oB = BSplineBasis(1, G_old, k, grid_range).b_splines(xe).squeeze(1)
            nB = BSplineBasis(1, G_new, k, grid_range).b_splines(xe).squeeze(1)
        return torch.linalg.lstsq(nB.float(), oB.float()).solution.float()

    class KASNLayer(nn.Module):

        def __init__(self, in_features, out_features, grid_size=5,
                     spline_order=KASN_SPLINE_ORDER,
                     base_activation=KASN_BASE_ACTIVATION,
                     grid_range=KASN_GRID_RANGE, use_residual=True):
            super().__init__()
            self.in_features, self.out_features = in_features, out_features
            self.use_residual = use_residual and (in_features == out_features)
            self.basis = BSplineBasis(in_features, grid_size, spline_order,
                                      grid_range)
            self.num_basis = grid_size + spline_order
            self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
            self.spline_weight = nn.Parameter(
                torch.Tensor(out_features, in_features, self.num_basis))
            self.w_b = nn.Parameter(torch.ones(1))
            self.w_s = nn.Parameter(torch.ones(1))
            self.w_b.requires_grad_(False); self.w_s.requires_grad_(False)
            self.base_activation = base_activation()
            self.grid_size = grid_size
            self.reset_parameters()

        def reset_parameters(self):
            nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
            nn.init.uniform_(self.spline_weight, -0.5 / self.grid_size,
                             0.5 / self.grid_size)
            with torch.no_grad():
                self.w_b.fill_(0.0); self.w_s.fill_(1.0)

        def set_transition(self, s):
            with torch.no_grad():
                self.w_b.fill_(1 - s); self.w_s.fill_(s)

        def forward(self, x):
            base = F.linear(self.base_activation(x), self.base_weight)
            spl = F.linear(self.basis(x).view(x.size(0), -1),
                           self.spline_weight.view(self.out_features, -1))
            out = self.w_b * base + self.w_s * spl
            return out + x if self.use_residual else out

        def l1_regularization_loss(self, s=KASN_L1_REG_SCALE):
            return s * torch.sum(torch.abs(self.spline_weight))

        def group_lasso_regularization_loss(self, gl):
            return gl * torch.norm(self.spline_weight, p=2, dim=2).sum()

        def count_active_edges(self, thr=KASN_ACTIVE_EDGE_THRESHOLD):
            with torch.no_grad():
                return int(torch.sum(
                    torch.norm(self.spline_weight, p=2, dim=2) > thr).item())

        def count_total_edges(self):
            return self.out_features * self.in_features

    class KASN(nn.Module):

        def __init__(self, input_dim, n_samples, gamma=0.3, kasn_width=None,
                     depth=None, zeta_delta=KASN_ZETA_DELTA,
                     dual_reg=KASN_DUAL_REG, kart_s=None, verbose=True):
            super().__init__()
            self.input_dim, self.n_samples = input_dim, n_samples
            self.gamma, self.zeta_delta, self.dual_reg = gamma, zeta_delta, dual_reg
            self.kart_s = kart_s if kart_s is not None else input_dim
            self.G = max(5, int(n_samples ** gamma))
            self.L = depth if depth is not None else max(3, int(np.log(n_samples)))
            self.W = kasn_width if kasn_width is not None else (2 * self.kart_s + 1)
            self.delta_n = (float('inf') if dual_reg else
                            max(5.0, np.log(n_samples), n_samples ** zeta_delta))
            if ADAPTIVE_GRID:
                cands = [max(5, int(s * self.G)) for s in KASN_GRID_PHASE_SCALES]
                cands.append(self.G)
                ph = []
                for g in cands:
                    if not ph or g > ph[-1]:
                        ph.append(g)
                self.G_phases = ph
            else:
                self.G_phases = [self.G]
            G0 = self.G_phases[0]
            if verbose:
                print(f"  KASN: L={self.L}  W={self.W}  G_n={self.G}  "
                      f"gamma={gamma}  d={input_dim}  r_n={self.r_n_framework()}"
                      f"  Delta_n={'dual' if dual_reg else f'{self.delta_n:.2f}'}")
            self.layers = nn.ModuleList()
            self.layers.append(KASNLayer(input_dim, self.W, grid_size=G0,
                                         use_residual=False))
            for _ in range(self.L - 2):
                self.layers.append(KASNLayer(self.W, self.W, grid_size=G0,
                                             use_residual=True))
            self.layers.append(KASNLayer(self.W, 1, grid_size=G0,
                                         use_residual=False))
            self.scaler_X = EmpiricalCDFTransformer()
            self.scaler_y = StandardScaler()

        def r_n_framework(self):
            return int((self.input_dim + 1) * self.W + (self.L - 2) * self.W ** 2)

        def s_0_bar(self):
            return float(self.kart_s * self.L)

        def compute_lambda_reg(self):
            return float(np.sqrt(np.log(self.L * self.W ** 2 + 1e-8)
                                 / self.n_samples))

        def penalty_rate_diagnostic(self):
            n = self.n_samples
            dn = n ** ((self.gamma - 1.0) / 2.0) * np.log(max(n, 3)) ** 2
            lhs = self.compute_lambda_reg() * self.s_0_bar()
            return {'lambda_n': self.compute_lambda_reg(),
                    's_0_bar': self.s_0_bar(), 'delta_n_rate': float(dn),
                    'lhs': float(lhs), 'rhs': float(dn ** 2),
                    'ratio': float(lhs / max(dn ** 2, 1e-300))}

        def _extend_to_G(self, G_new):
            for layer in self.layers:
                if layer.grid_size == G_new:
                    continue
                k, dev = layer.basis.spline_order, layer.spline_weight.device
                Mx = _oslo_projection_matrix(layer.grid_size, G_new, k,
                                             KASN_GRID_RANGE, OSLO_N_EVAL).to(dev)
                with torch.no_grad():
                    nw = layer.spline_weight.data @ Mx.t()
                layer.basis = BSplineBasis(layer.in_features, G_new, k,
                                           KASN_GRID_RANGE).to(dev)
                layer.num_basis, layer.grid_size = G_new + k, G_new
                layer.spline_weight = nn.Parameter(nw)

        def _rebuild_at_G(self, G_target):
            for layer in self.layers:
                if layer.grid_size == G_target:
                    continue
                k, dev = layer.basis.spline_order, layer.spline_weight.device
                layer.basis = BSplineBasis(layer.in_features, G_target, k,
                                           KASN_GRID_RANGE).to(dev)
                layer.num_basis = G_target + k
                layer.spline_weight = nn.Parameter(torch.zeros(
                    layer.out_features, layer.in_features, G_target + k, device=dev))
                layer.grid_size = G_target

        def l1_regularization_loss(self, s=KASN_L1_REG_SCALE):
            return sum(l.l1_regularization_loss(s) for l in self.layers)

        def group_lasso_regularization_loss(self, gl):
            return sum(l.group_lasso_regularization_loss(gl) for l in self.layers)

        def compute_current_total_l1(self):
            return sum(torch.norm(l.spline_weight, p=2, dim=2).sum().item()
                       for l in self.layers)

        def compute_delta_penalty(self, lam=1.0):
            if self.dual_reg:
                return torch.tensor(0.0, device=next(self.parameters()).device)
            tot = sum(torch.sum(torch.abs(l.spline_weight)) for l in self.layers)
            return lam * torch.clamp(tot - self.delta_n, min=0.0) ** 2

        def final_delta_projection(self, verbose=True):
            tot = self.compute_current_total_l1()
            if self.dual_reg:
                if verbose:
                    print(f"  Dual regularization: implied Delta_n = {tot:.4f}")
                self.delta_n = tot
                return
            if tot > self.delta_n:
                sc = self.delta_n / (tot + 1e-12)
                with torch.no_grad():
                    for l in self.layers:
                        l.spline_weight.data *= sc
                if verbose:
                    print(f"  Projected l1 {tot:.4f} -> {self.delta_n:.4f}")

        def _eff_delta(self):
            return (self.compute_current_total_l1()
                    if (self.dual_reg and not np.isfinite(self.delta_n))
                    else self.delta_n)

        def count_active_edges(self, thr=KASN_ACTIVE_EDGE_THRESHOLD):
            return sum(l.count_active_edges(thr) for l in self.layers)

        def count_total_edges(self):
            return sum(l.count_total_edges() for l in self.layers)

        def count_active_edges_delta_normalized(self):
            tot = self.count_total_edges()
            thr = self._eff_delta() / tot if tot > 0 else 0.0
            return self.count_active_edges(thr), thr, tot

        def count_active_edges_delta_over_r(self):
            r_n = self.r_n_framework()
            thr = self._eff_delta() / r_n if r_n > 0 else 0.0
            return self.count_active_edges(thr), thr, r_n

        def prune_edges(self, method='delta_normalized', val=None):
            if method == 'delta_normalized':
                tot = self.count_total_edges()
                thr = self._eff_delta() / tot if tot > 0 else 0.0
            elif method == 'delta_over_r':
                thr = self.count_active_edges_delta_over_r()[1]
            elif method == 'fixed':
                thr = val or KASN_ACTIVE_EDGE_THRESHOLD
            elif method == 'relative_fraction':
                with torch.no_grad():
                    mx = max(torch.norm(l.spline_weight, p=2, dim=2).max().item()
                             for l in self.layers)
                thr = (val or PRUNING_RELATIVE_FRACTION) * mx
            else:
                raise ValueError(f"Unknown pruning method: {method}")
            pruned = total = 0
            with torch.no_grad():
                for layer in self.layers:
                    en = torch.norm(layer.spline_weight, p=2, dim=2)
                    mask = en <= thr
                    layer.spline_weight.data[mask] = 0.0
                    pruned += int(mask.sum().item()); total += en.numel()
            return pruned, total

        def apply_post_training_pruning(self, method=PRUNING_THRESHOLD_METHOD,
                                        val=None, verbose=True):
            p, t = self.prune_edges(method, val)
            if verbose:
                print(f"  Post-training pruning ({method}): {p}/{t} edges "
                      f"zeroed ({p/t*100:.1f}%)")

        def set_prune_mask(self):
            nm = nt = 0
            with torch.no_grad():
                for layer in self.layers:
                    m = (torch.norm(layer.spline_weight, p=2, dim=2) == 0)
                    layer._prune_mask = m
                    nm += int(m.sum().item()); nt += m.numel()
            return nm, nt

        def apply_prune_mask(self):
            with torch.no_grad():
                for layer in self.layers:
                    m = getattr(layer, '_prune_mask', None)
                    if m is not None:
                        layer.spline_weight.data[m] = 0.0

        def masked_edge_counts(self):
            act = tot = 0
            with torch.no_grad():
                for layer in self.layers:
                    m = getattr(layer, '_prune_mask', None)
                    if m is None:
                        m = (torch.norm(layer.spline_weight, p=2, dim=2) == 0)
                    act += int((~m).sum().item()); tot += m.numel()
            return act, tot

        def sparsity_report(self, C_s_n=KASN_C_S_N):
            tot = self.count_total_edges()
            ae_dn, thr_dn, _ = self.count_active_edges_delta_normalized()
            ae_dr, thr_dr, r_n = self.count_active_edges_delta_over_r()
            cap = math.ceil(C_s_n * self.s_0_bar())
            return {'active_edges_delta': ae_dn, 'threshold_delta': thr_dn,
                    'active_edges_delta_over_r': ae_dr,
                    'threshold_delta_over_r': thr_dr, 'total_edges': tot,
                    'r_n': r_n, 'delta_n': self._eff_delta(),
                    'delta_sparsity': 1 - ae_dn / tot if tot else np.nan,
                    'C_s_n': float(C_s_n), 's_n_cap': cap,
                    'cap_respected': bool(ae_dr <= cap)}

        def set_transition(self, s):
            for l in self.layers:
                l.set_transition(s)

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

        def fit(self, X_tr_np, y_tr_np, X_va_np, y_va_np, epochs=1000, lr=1e-3,
                batch_size=512, patience=50, weight_decay=KASN_WEIGHT_DECAY,
                l1_reg_scale=KASN_L1_REG_SCALE, group_lasso_reg_scale=1e-3,
                resume_scalers=False, verbose=True, tag=''):
            y_tr_np = y_tr_np.reshape(-1, 1) if y_tr_np.ndim == 1 else y_tr_np
            y_va_np = y_va_np.reshape(-1, 1) if y_va_np.ndim == 1 else y_va_np
            if not resume_scalers:
                self.scaler_X.fit(X_tr_np); self.scaler_y.fit(y_tr_np)
            X_tr = torch.tensor(self.scaler_X.transform(X_tr_np), dtype=torch.float32)
            y_tr = torch.tensor(self.scaler_y.transform(y_tr_np), dtype=torch.float32)
            X_va = torch.tensor(self.scaler_X.transform(X_va_np), dtype=torch.float32)
            y_va = torch.tensor(self.scaler_y.transform(y_va_np), dtype=torch.float32)

            pin = (DEVICE.type == 'cuda')
            loader = DataLoader(TensorDataset(X_tr, y_tr),
                                batch_size=min(batch_size, len(X_tr)),
                                shuffle=True, drop_last=False, pin_memory=pin)
            self.to(DEVICE)
            Xv, yv = X_va.to(DEVICE), y_va.to(DEVICE)
            opt = torch.optim.AdamW(self.parameters(), lr=lr,
                                    weight_decay=weight_decay)
            sched = torch.optim.lr_scheduler.StepLR(opt, step_size=500, gamma=0.5)
            crit = nn.MSELoss()
            amp = USE_AMP and (DEVICE.type == 'cuda')
            gs = torch.amp.GradScaler('cuda', enabled=amp)
            lam = self.compute_lambda_reg()

            best, bstate, bepoch, noimp = float('inf'), None, 0, 0
            gbest = {'val': float('inf'), 'state': None, 'epoch': 0,
                     'G': self.layers[0].grid_size}
            tr_hist, va_hist, epoch_log = [], [], []
            milestones = ({int(KASN_GRID_PHASE_FRACS[i] * epochs): self.G_phases[i + 1]
                           for i in range(min(len(self.G_phases) - 1,
                                              len(KASN_GRID_PHASE_FRACS)))}
                          if (ADAPTIVE_GRID and len(self.G_phases) > 1) else {})

            for epoch in range(epochs):
                if epoch in milestones:
                    self._extend_to_G(milestones[epoch])
                    ph = sum(1 for e in milestones if e <= epoch)
                    opt = torch.optim.AdamW(self.parameters(), lr=lr * 0.5 ** ph,
                                            weight_decay=weight_decay)
                    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=500,
                                                            gamma=0.5)
                    gs = torch.amp.GradScaler('cuda', enabled=amp)
                    best, bstate, noimp = float('inf'), None, 0

                self.train(); self.set_transition(1.0)
                emse = 0.0
                for Xb, yb in loader:
                    Xb = Xb.to(DEVICE, non_blocking=True)
                    yb = yb.to(DEVICE, non_blocking=True)
                    opt.zero_grad(set_to_none=True)
                    with torch.amp.autocast(device_type=DEVICE.type, enabled=amp):
                        mse = crit(self(Xb), yb)
                        gl = self.group_lasso_regularization_loss(group_lasso_reg_scale)
                        l1 = self.l1_regularization_loss(l1_reg_scale)
                        tot = mse + lam * (gl + l1) + self.compute_delta_penalty()
                    gs.scale(tot).backward()
                    gs.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(self.parameters(), 500.0)
                    gs.step(opt); gs.update()
                    self.apply_prune_mask()
                    emse += mse.item() * Xb.size(0)
                sched.step()
                avg = emse / len(X_tr); tr_hist.append(avg)

                self.eval()
                with torch.no_grad():
                    parts = []
                    for i in range(0, Xv.shape[0], INFERENCE_CHUNK_SIZE):
                        with torch.amp.autocast(device_type=DEVICE.type, enabled=amp):
                            parts.append(self(Xv[i:i + INFERENCE_CHUNK_SIZE]))
                    vloss = crit(torch.cat(parts), yv).item()
                va_hist.append(vloss)

                if vloss < best:
                    best, bepoch, noimp = vloss, epoch, 0
                    bstate = copy.deepcopy(self.state_dict())
                else:
                    noimp += 1
                if vloss < gbest['val']:
                    gbest = {'val': vloss, 'epoch': epoch,
                             'G': self.layers[0].grid_size,
                             'state': copy.deepcopy(self.state_dict())}

                if epoch % EPOCH_LOG_EVERY == 0 or epoch == epochs - 1:
                    ae, _, te = self.count_active_edges_delta_normalized()
                    ds = 1 - ae / te if te else np.nan
                    mon = _eval_monitor_str(self)
                    if verbose:
                        print(f"  {tag}Epoch {epoch:4d}/{epochs}: "
                              f"Train={avg:.6f}, Val={vloss:.6f}, "
                              f"BestVal={best:.6f} (ep {bepoch}), "
                              f"ActiveEdges={ae}/{te}, D-Sparsity={ds:.4f}, "
                              f"LR={sched.get_last_lr()[0]:.6f}{mon}")
                    rec = {'epoch': epoch, 'train_loss': avg, 'val_mse': vloss,
                           'best_val_mse': best, 'best_epoch': bepoch,
                           'active_edges': ae, 'total_edges': te,
                           'delta_sparsity': round(float(ds), 6),
                           'lr': sched.get_last_lr()[0],
                           'epochs_no_improve': noimp,
                           'grid_size': self.layers[0].grid_size}
                    if _EVAL_MONITOR is not None:
                        for sp in ('val', 'test'):
                            Xm, ym = _EVAL_MONITOR[f'X_{sp}'], _EVAL_MONITOR[f'y_{sp}']
                            rec[f'r2_{sp}'] = (_r2(ym, self.predict(Xm))
                                               if Xm is not None else np.nan)
                    epoch_log.append(rec)

                final_phase = (not milestones or epoch >= max(milestones))
                if patience is not None and noimp >= patience and final_phase:
                    if verbose:
                        print(f"  {tag}Early stopping at epoch {epoch} "
                              f"(best {bepoch}, val {best:.6f})")
                    break

            if GLOBAL_BEST_RESTORE and gbest['state'] is not None:
                if gbest['G'] != self.layers[0].grid_size:
                    self._rebuild_at_G(gbest['G']); self.G = gbest['G']
                self.load_state_dict(gbest['state']); self.to(DEVICE)
                best, bepoch = gbest['val'], gbest['epoch']
                if verbose:
                    print(f"  {tag}Restored global best from epoch {bepoch} "
                          f"(val {best:.6f}, PRE-PRUNING)")
            elif bstate is not None:
                self.load_state_dict(bstate)
            return tr_hist, va_hist, best, epoch_log

        def predict(self, X_np):
            self.eval(); self.set_transition(1.0); self.to(DEVICE)
            amp = USE_AMP and (DEVICE.type == 'cuda')
            X_t = torch.tensor(self.scaler_X.transform(X_np), dtype=torch.float32)
            parts = []
            with torch.inference_mode():
                for i in range(0, X_t.shape[0], INFERENCE_CHUNK_SIZE):
                    ch = X_t[i:i + INFERENCE_CHUNK_SIZE].to(DEVICE)
                    with torch.amp.autocast(device_type=DEVICE.type, enabled=amp):
                        parts.append(self(ch).float().cpu())
            return self.scaler_y.inverse_transform(torch.cat(parts).numpy()).flatten()


    class FFANNNorets(nn.Module):

        def __init__(self, input_dim, hidden=FFANN_HIDDEN,
                     c_weight=FFANN_C_WEIGHT, c_out=FFANN_C_OUT, verbose=True):
            super().__init__()
            self.input_dim, self.hidden = input_dim, hidden
            self.c_weight, self.c_out = float(c_weight), float(c_out)
            self.fc1 = nn.Linear(input_dim, hidden)
            self.fc2 = nn.Linear(hidden, 1, bias=True)
            nn.init.xavier_uniform_(self.fc1.weight); nn.init.zeros_(self.fc1.bias)
            nn.init.xavier_uniform_(self.fc2.weight)
            self.act = nn.Sigmoid()
            self.scaler_X = EmpiricalCDFTransformer()
            self.scaler_y = StandardScaler()
            if verbose:
                print(f"  FFANN (Norets 2012): {input_dim}-{hidden}-1 logistic, "
                      f"White (1990) weight bounds {c_weight}/{c_out}")

        def _constrained(self):
            wl = self.fc1.weight.abs().sum() + 1e-8
            sc = torch.clamp(self.hidden * self.c_weight / wl, max=1.0)
            w1, b1 = self.fc1.weight * sc, self.fc1.bias * sc
            ol = self.fc2.weight.abs().sum() + 1e-8
            w2 = self.fc2.weight * torch.clamp(self.c_out / ol, max=1.0)
            return w1, b1, w2

        def forward(self, x):
            w1, b1, w2 = self._constrained()
            return F.linear(self.act(F.linear(x, w1, b1)), w2)

        def fit(self, X_tr_np, y_tr_np, X_va_np, y_va_np, epochs=FFANN_EPOCHS,
                lr=FFANN_LR, batch_size=FFANN_BATCH, patience=FFANN_PATIENCE,
                verbose=True):
            y_tr_np = y_tr_np.reshape(-1, 1) if y_tr_np.ndim == 1 else y_tr_np
            y_va_np = y_va_np.reshape(-1, 1) if y_va_np.ndim == 1 else y_va_np
            self.scaler_X.fit(X_tr_np); self.scaler_y.fit(y_tr_np)
            X_tr = torch.tensor(self.scaler_X.transform(X_tr_np), dtype=torch.float32)
            y_tr = torch.tensor(self.scaler_y.transform(y_tr_np), dtype=torch.float32)
            X_va = torch.tensor(self.scaler_X.transform(X_va_np), dtype=torch.float32)
            y_va = torch.tensor(self.scaler_y.transform(y_va_np), dtype=torch.float32)
            loader = DataLoader(TensorDataset(X_tr, y_tr),
                                batch_size=min(batch_size, len(X_tr)), shuffle=True)
            self.to(DEVICE)
            Xv, yv = X_va.to(DEVICE), y_va.to(DEVICE)
            opt = torch.optim.Adam(self.parameters(), lr=lr)
            crit = nn.MSELoss()
            best, bstate, bepoch, noimp = float('inf'), None, 0, 0
            for epoch in range(epochs):
                self.train()
                for Xb, yb in loader:
                    Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                    opt.zero_grad(set_to_none=True)
                    crit(self(Xb), yb).backward()
                    opt.step()
                self.eval()
                with torch.no_grad():
                    vloss = crit(self(Xv), yv).item()
                if vloss < best:
                    best, bepoch, noimp = vloss, epoch, 0
                    bstate = copy.deepcopy(self.state_dict())
                else:
                    noimp += 1
                if verbose and epoch % 50 == 0:
                    print(f"  FFANN epoch {epoch:4d}/{epochs}: val={vloss:.6f} "
                          f"best={best:.6f} (ep {bepoch})")
                if patience is not None and noimp >= patience:
                    if verbose:
                        print(f"  FFANN early stopping at epoch {epoch} "
                              f"(best {bepoch}, val {best:.6f})")
                    break
            if bstate is not None:
                self.load_state_dict(bstate)
            return best

        def set_transition(self, s):
            return None

        def predict(self, X_np):
            self.eval(); self.to(DEVICE)
            X_t = torch.tensor(self.scaler_X.transform(X_np), dtype=torch.float32)
            parts = []
            with torch.inference_mode():
                for i in range(0, X_t.shape[0], INFERENCE_CHUNK_SIZE):
                    parts.append(self(X_t[i:i + INFERENCE_CHUNK_SIZE].to(DEVICE)).cpu())
            return self.scaler_y.inverse_transform(torch.cat(parts).numpy()).flatten()



def _scaled_val_loss(model, X_val, y_val):
    yv = np.asarray(y_val, dtype=np.float64).reshape(-1, 1)
    a = model.scaler_y.transform(yv).ravel()
    b = model.scaler_y.transform(np.asarray(model.predict(X_val),
                                            dtype=np.float64).reshape(-1, 1)).ravel()
    return float(np.mean((a - b) ** 2))


def prune_and_finetune(model, X_tr, y_tr, X_va, y_va, lr, batch_size,
                       group_lasso_reg_scale, label='', verbose=True):
    val_pre = _scaled_val_loss(model, X_va, y_va)
    if not POST_TRAINING_PRUNING:
        return val_pre, val_pre, val_pre
    model.apply_post_training_pruning(verbose=verbose)
    val_post = _scaled_val_loss(model, X_va, y_va)
    val_ft = val_post
    if POST_PRUNE_FINETUNE_EPOCHS > 0:
        nm, nt = model.set_prune_mask()
        if verbose:
            print(f"  {label}Fine-tuning {POST_PRUNE_FINETUNE_EPOCHS} epochs, "
                  f"{nm}/{nt} edges frozen at zero ({100*nm/nt:.1f}%)")
        model.fit(X_tr, y_tr, X_va, y_va, epochs=POST_PRUNE_FINETUNE_EPOCHS,
                  lr=lr * POST_PRUNE_FINETUNE_LR_SCALE, batch_size=batch_size,
                  patience=POST_PRUNE_FINETUNE_PATIENCE,
                  group_lasso_reg_scale=group_lasso_reg_scale,
                  resume_scalers=True, verbose=False)
        model.apply_prune_mask()
        val_ft = _scaled_val_loss(model, X_va, y_va)
        act, tot = model.masked_edge_counts()
        assert tot - act == nm, (
            f"prune mask not respected: {tot-act} zeroed vs {nm} masked")
        if verbose:
            rec = 100 * (val_post - val_ft) / max(val_post - val_pre, 1e-30)
            print(f"  {label}val {val_pre:.6f} pre -> {val_post:.6f} pruned "
                  f"-> {val_ft:.6f} fine-tuned (recovered {rec:.0f}% of the "
                  f"pruning loss), active {act}/{tot}")
    return val_pre, val_post, val_ft


def select_kasn_hyperparams(X_tr, y_tr, X_va, y_va, n_samples, input_dim,
                            depth=4, batch_size=512, tuning_epochs=60,
                            tuning_patience=12, verbose=True):
    t0, log = time.time(), []
    cur = {'gamma': KASN_GAMMA_GRID[0], 'lr': KASN_LR_GRID[0],
           'gl': KASN_GL_GRID[0], 'width': KASN_WIDTH_GRID[0]}
    n_cands = sum(len(g) for g in (KASN_GAMMA_GRID, KASN_LR_GRID,
                                   KASN_GL_GRID, KASN_WIDTH_GRID))
    if verbose:
        scored = ("PRUNED + FINE-TUNED"
                  if PRUNE_AWARE_SELECTION and POST_TRAINING_PRUNING else "unpruned")
        print(f"\n{'='*74}")
        print(f"KASN HYPERPARAMETER SELECTION  ({n_cands} candidates, "
              f"{tuning_epochs} epochs, patience {tuning_patience})")
        print(f"  Greedy order: gamma -> lr -> group lasso -> width")
        print(f"  Scoring the {scored} estimator, the object that gets reported")
        print(f"{'='*74}")

    def _run(g, lr, gl, w, phase):
        cand = KASN(input_dim=input_dim, n_samples=n_samples, gamma=g,
                    kasn_width=w, depth=depth, verbose=False)
        _, _, vl_un, _ = cand.fit(X_tr, y_tr, X_va, y_va, epochs=tuning_epochs,
                                  lr=lr, batch_size=batch_size,
                                  patience=tuning_patience,
                                  group_lasso_reg_scale=gl, verbose=False)
        vl, vl_pr = vl_un, np.nan
        if PRUNE_AWARE_SELECTION and POST_TRAINING_PRUNING:
            _, vl_pr, vl = prune_and_finetune(cand, X_tr, y_tr, X_va, y_va,
                                              lr=lr, batch_size=batch_size,
                                              group_lasso_reg_scale=gl,
                                              verbose=False)
        ae, te = cand.masked_edge_counts()
        log.append({'phase': phase, 'gamma': g, 'lr': lr, 'gl': gl, 'width': w,
                    'G_n': cand.G, 'r_n': cand.r_n_framework(), 'val_loss': vl,
                    'val_loss_unpruned': vl_un, 'val_loss_pruned': vl_pr,
                    'post_prune_sparsity': 1 - ae / te if te else np.nan,
                    'prune_aware': bool(PRUNE_AWARE_SELECTION),
                    'is_selected': False})
        del cand; gc.collect()
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()
        return vl

    for i, (name, grid) in enumerate([('gamma', KASN_GAMMA_GRID),
                                      ('lr', KASN_LR_GRID),
                                      ('gl', KASN_GL_GRID),
                                      ('width', KASN_WIDTH_GRID)], start=1):
        if len(grid) <= 1:
            cur[name] = grid[0]
            if verbose:
                print(f"\n  Phase {i}/4 - {name} fixed at {grid[0]}")
            continue
        if verbose:
            print(f"\n  Phase {i}/4 - {name}")
        best = float('inf')
        for v in grid:
            trial = dict(cur); trial[name] = v
            vl = _run(trial['gamma'], trial['lr'], trial['gl'], trial['width'],
                      phase=f"{i}_{name}")
            mk = "  <- best" if vl < best else ""
            if verbose:
                print(f"    {name}={v}  val={vl:.6f}{mk}")
            if vl < best:
                best, cur[name] = vl, v
        if verbose:
            print(f"  -> selected {name} = {cur[name]}")

    for e in log:
        e['is_selected'] = all(abs(float(e[k]) - float(cur[k])) < 1e-12
                               for k in ('gamma', 'lr', 'gl', 'width'))
    sel_t = time.time() - t0
    if verbose:
        print(f"\n  Selected: gamma={cur['gamma']}, lr={cur['lr']:.0e}, "
              f"gl={cur['gl']:.0e}, W={cur['width']}")
        print(f"  Selection time: {sel_t:.1f}s\n{'='*74}\n")
    return cur['gamma'], cur['lr'], cur['gl'], cur['width'], sel_t, log


def build_kasn_training_log(hp_log, epoch_log, gamma, lr, bs, gl, width,
                            model, n_train):
    rows = [{'log_type': 'hp_selection', **e} for e in (hp_log or [])]
    rows += [{'log_type': 'training_epoch', **e, 'selected_gamma': gamma,
              'selected_lr': lr, 'selected_bs': bs, 'selected_gl': gl,
              'selected_width': width} for e in (epoch_log or [])]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    pr = model.penalty_rate_diagnostic()
    df['dgp'], df['n_train'] = DGP, n_train
    df['kasn_depth'], df['kasn_spline_order'] = model.L, KASN_SPLINE_ORDER
    df['dual_reg'], df['implied_delta_n'] = KASN_DUAL_REG, model.delta_n
    df['zeta_delta'], df['lambda_n'] = KASN_ZETA_DELTA, pr['lambda_n']
    df['penalty_rate_ratio'] = pr['ratio']
    df['seed'], df['run_id'] = SEED, RUN_ID
    df['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
    return df



def _newey_west_variance(psi, bandwidth=None):
    psi = np.asarray(psi, dtype=np.float64)
    n = psi.size
    bw = int(np.floor(n ** (1.0 / 3.0))) if bandwidth is None else bandwidth
    pc = psi - psi.mean()
    var = float(np.mean(pc ** 2))
    for j in range(1, bw + 1):
        var += 2.0 * (1.0 - j / (bw + 1.0)) * float(np.mean(pc[j:] * pc[:-j]))
    return max(var, 0.0) / n


def _cos_basis(Xt, cols, K):
    n = Xt.shape[0]
    k = np.arange(1, K + 1)[None, :]
    blocks = [np.ones((n, 1))]
    for c in cols:
        blocks.append(np.sqrt(2.0) * np.cos(k * np.pi * Xt[:, c][:, None]))
    return np.hstack(blocks)


def _cos_basis_deriv(Xt, cols, K, j):
    n = Xt.shape[0]
    k = np.arange(1, K + 1)[None, :]
    blocks = [np.zeros((n, 1))]
    for c in cols:
        blocks.append(-np.sqrt(2.0) * k * np.pi
                      * np.sin(k * np.pi * Xt[:, c][:, None])
                      if c == j else np.zeros((n, K)))
    return np.hstack(blocks)


def analytic_riesz(design, r, coord):
    name = design.names[coord]
    a, b = design.shape(name)
    if name in ('eta1', 'eta2'):
        raise NotImplementedError(
            "analytic representer assumes independent design coordinates; "
            "eta1 and eta2 are Dirichlet and jointly dependent")
    lo, hi, kind = design.spec[name]
    span = hi - lo
    if kind == 'unif':
        return np.zeros_like(np.asarray(r, dtype=np.float64))
    r = np.clip(np.asarray(r, dtype=np.float64), 1e-10, 1 - 1e-10)
    u = np.clip(scipy.stats.beta.ppf(r, a, b), 1e-12, 1 - 1e-12)
    return -((a - 1.0) / u - (b - 1.0) / (1.0 - u)) / span


def riesz_second_moment_finite(design, coord=None) -> bool:
    if coord is None:
        return all(design.shape(n)[0] > 2.0 and design.shape(n)[1] > 2.0
                   for n in design.names if n not in ('eta1', 'eta2'))
    a, b = design.shape(coord)
    return bool(a > 2.0 and b > 2.0)


def _fit_riesz_alpha(Xt_fit, cols, K, j, w_fit):
    B = _cos_basis(Xt_fit, cols, K)
    dB = _cos_basis_deriv(Xt_fit, cols, K, j)
    G = B.T @ B / B.shape[0]
    ridge = max(AD_RIESZ_RIDGE,
                AD_RIESZ_RIDGE_SCALE * float(np.trace(G)) / G.shape[0])
    return np.linalg.solve(G + ridge * np.eye(G.shape[0]),
                           (w_fit[:, None] * dB).mean(axis=0))


def _density_admissible(col, min_frac=AD_MIN_DISTINCT_FRAC):
    v = np.asarray(col, dtype=np.float64)
    nd = int(np.unique(v).size)
    return bool(nd >= max(20, min_frac * v.size)), nd


def compute_average_derivatives(model, X_eval, X_fit, design, dgp,
                                y_eval=None, y_eval_pred=None,
                                nw_bandwidth=AD_NW_BANDWIDTH, verbose=True):
    names = list(dgp.feature_names)
    labels = ad_labels_for(dgp)
    idx = [names.index(l) for l in labels]
    lab = [names[i] for i in idx]
    if not idx:
        return {}
    kind = {l: ('state' if l in AD_STATE_LABELS_BY_DGP.get(dgp.name, [])
                else 'parameter') for l in lab}
    if verbose:
        print(f"\n{'='*74}")
        print("WEIGHTED AVERAGE DERIVATIVE  (Example 2 / Proposition 2)")
        print(f"  features: {lab}   n_eval={len(X_eval):,}   "
              f"weight: analytic design marginals")
        print(f"  state coordinates: "
              f"{[l for l in lab if kind[l]=='state'] or 'none'}   "
              f"parameter coordinates: "
              f"{[l for l in lab if kind[l]=='parameter'] or 'none'}")
        print(f"{'='*74}")

    model.eval(); model.set_transition(1.0); model.to(DEVICE)
    amp = USE_AMP and (DEVICE.type == 'cuda')
    Xs = model.scaler_X.transform(X_eval).astype(np.float32)
    n_ev = Xs.shape[0]
    grads = []
    for st in range(0, n_ev, AD_GRAD_CHUNK_SIZE):
        Xc = torch.tensor(Xs[st:st + AD_GRAD_CHUNK_SIZE], dtype=torch.float32,
                          requires_grad=True, device=DEVICE)
        with torch.amp.autocast(device_type=DEVICE.type, enabled=amp):
            yc = model(Xc)
        yc.sum().backward()
        grads.append(Xc.grad.detach().float().cpu().numpy())
        del Xc, yc
    grad = np.concatenate(grads, axis=0) * float(model.scaler_y.scale_[0])
    Xt_ev = Xs.astype(np.float64)
    Xt_fit = model.scaler_X.transform(X_fit).astype(np.float64)

    resid = theta_hat = None
    if AD_RIESZ_CORRECTION and y_eval is not None and y_eval_pred is not None:
        theta_hat = np.asarray(y_eval_pred, dtype=np.float64).ravel()
        resid = np.asarray(y_eval, dtype=np.float64).ravel() - theta_hat
        if np.allclose(resid, 0.0):
            if verbose:
                print("  Residuals are identically zero (noiseless targets), so "
                      "the U_t v* correction vanishes by construction.")
            resid = None

    bw = nw_bandwidth or int(np.floor(n_ev ** (1.0 / 3.0)))
    out = {}
    for j, name in zip(idx, lab):
        ok, nd = _density_admissible(X_fit[:, j])
        if not ok:
            if verbose:
                print(f"\n  [{j}] {name}: SKIPPED, {nd} distinct design values, "
                      f"so the marginal has atoms and w_j is undefined.")
            out[name] = {'feature': name, 'feature_index': j,
                         'coordinate_kind': kind[name], 'mu_hat': np.nan,
                         'admissible': False, 'n_distinct': nd}
            continue
        w_ev = design.weight_on_ranks(Xt_ev[:, j], coord=j)
        w_fit = design.weight_on_ranks(Xt_fit[:, j], coord=j)
        e1 = design.satisfies_E1(j)
        psi_j = w_ev * grad[:, j]
        mu = float(psi_j.mean())
        v_ev, gap, gap_abs, n_cap = np.zeros(n_ev), np.nan, np.nan, 0
        if resid is not None:
            if AD_RIESZ_METHOD == 'analytic':
                v_ev = analytic_riesz(design, Xt_ev[:, j], j)
                rms_th = design.riesz_rms(j)
                if AD_RIESZ_CAP_MULT is not None and np.isfinite(rms_th):
                    cap = AD_RIESZ_CAP_MULT * rms_th
                    n_cap = int(np.sum(np.abs(v_ev) > cap))
                    v_ev = np.clip(v_ev, -cap, cap)
                elif AD_RIESZ_WINSOR is not None:
                    cap = float(np.quantile(np.abs(v_ev), AD_RIESZ_WINSOR))
                    n_cap = int(np.sum(np.abs(v_ev) > cap))
                    v_ev = np.clip(v_ev, -cap, cap)
                else:
                    n_cap = 0
            else:
                a = _fit_riesz_alpha(Xt_fit, idx, AD_RIESZ_N_COS, j, w_fit)
                v_ev = _cos_basis(Xt_ev, idx, AD_RIESZ_N_COS) @ a
            gap_abs = abs(float(np.mean(v_ev * theta_hat)) - mu)
            se_scale = float(psi_j.std()) / max(np.sqrt(n_ev), 1.0)
            gap = gap_abs / max(abs(mu) + se_scale, 1e-12)
        psi = psi_j - mu + (resid * v_ev if resid is not None else 0.0)
        se = float(np.sqrt(_newey_west_variance(psi, bw)))
        se_n = float(np.sqrt(_newey_west_variance(psi_j, bw)))
        out[name] = {'feature': name, 'feature_index': j,
                     'coordinate_kind': kind[name], 'mu_hat': mu,
                     'se_hat': se, 'se_naive': se_n, 'ci_lower': mu - 1.96 * se,
                     'ci_upper': mu + 1.96 * se, 'riesz_gap': gap,
                     'nw_bandwidth': bw, 'n_distinct': nd, 'admissible': True,
                     'boundary_condition_E1': e1,
                     'riesz_gap_abs': gap_abs,
                     'riesz_method': AD_RIESZ_METHOD,
                     'riesz_L2_finite': riesz_second_moment_finite(design, j),
                     'riesz_v_rms': float(np.sqrt(np.mean(v_ev ** 2))),
                     'riesz_v_rms_theory': design.riesz_rms(j),
                     'riesz_n_capped': (n_cap if resid is not None else 0),
                     'mean_weight': float(w_ev.mean())}
        if verbose:
            flag = "" if e1 else "   [(E.1) fails: design does not vanish at the boundary]"
            print(f"\n  [{j}] {name}  ({kind[name]}){flag}")
            print(f"    Gamma_hat = {mu:+.6f}   (original covariate scale)")
            print(f"    SE        = {se:.6f} corrected,  {se_n:.6f} naive")
            print(f"    95% CI    = [{mu - 1.96*se:+.6f}, {mu + 1.96*se:+.6f}]")
            if np.isfinite(gap):
                print(f"    Riesz moment gap = {gap:.4f}")
    if verbose:
        print(f"\n{'='*74}\n")
    return out


def counterfactual_difference(model, design, dgp, X_eval, y_eval=None,
                              param=CF_PARAM, tau=CF_TAU, center=CF_CENTER,
                              nw_bandwidth=AD_NW_BANDWIDTH, verbose=True):
    names = list(dgp.feature_names)
    if param not in names:
        if verbose:
            print(f"  [CF] '{param}' is not an input coordinate, skipping.")
        return {}
    j = names.index(param)
    lo, hi, kindj = design.spec[param]
    c = 0.5 * (lo + hi) if center is None else float(center)
    if not (lo < c < hi):
        raise ValueError("the counterfactual target must lie inside the support")
    if not (0.0 < tau < 1.0):
        raise ValueError("tau must lie strictly between zero and one")

    a = np.asarray(X_eval[:, j], dtype=np.float64)
    a_inv = (a - tau * c) / (1.0 - tau)

    def _dens(v):
        v = np.asarray(v, dtype=np.float64)
        if kindj == 'unif':
            return np.where((v >= lo) & (v <= hi), 1.0 / (hi - lo), 0.0)
        u = (v - lo) / (hi - lo)
        a_n, b_n = design.shape(param)
        d = scipy.stats.beta.pdf(np.clip(u, 0.0, 1.0), a_n, b_n)
        return np.where((u >= 0.0) & (u <= 1.0), d / (hi - lo), 0.0)

    f_mu = _dens(a)
    f_nu = np.where((a_inv >= lo) & (a_inv <= hi),
                    _dens(a_inv) / (1.0 - tau), 0.0)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(f_mu > 1e-12, f_nu / np.maximum(f_mu, 1e-12), 0.0)
    n_clip = int(np.sum(ratio > CF_OMEGA_CLIP))
    ratio = np.clip(ratio, 0.0, CF_OMEGA_CLIP)
    omega = ratio - 1.0

    F_hat = model.predict(X_eval)
    X_cf = X_eval.copy()
    X_cf[:, j] = (1.0 - tau) * a + tau * c
    diff = model.predict(X_cf) - F_hat

    gamma = float(np.mean(diff))
    gamma_ratio = float(np.mean(omega * F_hat))

    if y_eval is not None:
        U = np.asarray(y_eval, dtype=np.float64).ravel() - F_hat
        psi = diff - gamma + U * omega
        corrected = True
    else:
        psi = diff - gamma
        corrected = False
    bw = nw_bandwidth or int(np.floor(len(psi) ** (1.0 / 3.0)))
    se = float(np.sqrt(_newey_west_variance(psi, bw)))
    se_naive = float(np.sqrt(_newey_west_variance(diff - gamma, bw)))

    out = {'parameter': param, 'tau': tau, 'target_c': c,
           'support_lo': lo, 'support_hi': hi, 'design_kind': kindj,
           'gamma_cf': gamma, 'se_hat': se, 'se_naive': se_naive,
           'ci_lower': gamma - 1.96 * se, 'ci_upper': gamma + 1.96 * se,
           'first_stage_corrected': corrected,
           'gamma_ratio_form': gamma_ratio,
           'omega_mean': float(np.mean(omega)),
           'omega_max_abs': float(np.max(np.abs(omega))),
           'n_omega_clipped': n_clip, 'nw_bandwidth': bw,
           'n_eval': int(len(a))}
    if verbose:
        print(f"\n{'='*74}")
        print("FIXED COUNTERFACTUAL DIFFERENCE  (linear functional, Theorem 4)")
        print(f"{'='*74}")
        print(f"  policy: {param} -> (1-{tau}) * {param} + {tau} * {c:.4f}   "
              f"on support [{lo:.4f}, {hi:.4f}]")
        print(f"  Gamma_cf  = {gamma:+.6f}   (direct evaluation at T(X))")
        print(f"  SE        = {se:.6f} corrected,  {se_naive:.6f} naive")
        print(f"  95% CI    = [{gamma - 1.96*se:+.6f}, {gamma + 1.96*se:+.6f}]")
        print(f"  cross-check: density-ratio form = {gamma_ratio:+.6f}   "
              f"(same estimand, much noisier at this n)")
        print(f"  omega: mean {out['omega_mean']:+.4f} (population value zero), "
              f"max abs {out['omega_max_abs']:.3f}"
              + (f", {n_clip} clipped at {CF_OMEGA_CLIP}" if n_clip else ""))
        print(f"  The representer is omega itself, so K.5 holds trivially and "
              f"no sieve approximation enters the correction.")
        print(f"{'='*74}\n")
    return out


def average_ccp_effect(model, dgp, X_eval, design, verbose=True):
    Xs = model.scaler_X.transform(X_eval).astype(np.float32)
    model.eval(); model.set_transition(1.0); model.to(DEVICE)
    Xc = torch.tensor(Xs, dtype=torch.float32, requires_grad=True, device=DEVICE)
    model(Xc).sum().backward()
    grad = Xc.grad.detach().float().cpu().numpy() * float(model.scaler_y.scale_[0])
    dF = design.weight_on_ranks(Xs.astype(np.float64)[:, 0], coord=0) * grad[:, 0]
    F_hat = model.predict(X_eval)
    names = list(dgp.feature_names)
    a1 = X_eval[:, names.index('alpha1')]
    a2 = X_eval[:, names.index('alpha2')]
    x = X_eval[:, 0]
    if dgp.name == 'ev_iid':
        g = a2 - a1 * x - F_hat
        dens = np.exp(-np.abs(g)) / (1.0 + np.exp(-np.abs(g))) ** 2
    else:
        xi = X_eval[:, names.index('xi')]
        dens = _norm.pdf((a2 - a1 * x - xi - F_hat) / dgp.sigma_eta) / dgp.sigma_eta
    eff = -dens * (a1 + dF)
    if verbose:
        print(f"  Average marginal effect on P(replace): {eff.mean():+.6e}   "
              f"[diagnostic, nonlinear functional, no interval claimed]")
    return {'mean_ccp_effect': float(eff.mean()),
            'sd_ccp_effect': float(eff.std())}



def posterior_summary(chain_e, chain_k, truth, param_names):
    rows = []
    for name in param_names:
        if name.startswith('eta'):
            k = int(name[-1]) - 1
            ve, vk, tv = chain_e['eta'][:, k], chain_k['eta'][:, k], truth['eta'][k]
        else:
            if name not in chain_e:
                continue
            ve, vk, tv = chain_e[name], chain_k[name], truth[name]
        em, es, km, ks = ve.mean(), ve.std(), vk.mean(), vk.std()
        rows.append({'parameter': name, 'truth': tv, 'exact_mean': em,
                     'exact_sd': es, 'kasn_mean': km, 'kasn_sd': ks,
                     'abs_diff_over_sd': abs(km - em) / (es + 1e-12),
                     'exact_bias': em - tv, 'kasn_bias': km - tv,
                     'exact_q025': np.quantile(ve, 0.025),
                     'exact_q975': np.quantile(ve, 0.975),
                     'kasn_q025': np.quantile(vk, 0.025),
                     'kasn_q975': np.quantile(vk, 0.975)})
    return pd.DataFrame(rows)


def print_posterior_summary(df):
    print("\n" + "=" * 92)
    print("POSTERIOR SUMMARIES   (|Delta mean(KASN, exact)| / sd_exact, last column)")
    print("=" * 92)
    print(f"{'param':<10}{'truth':>11}{'exact mean':>13}{'exact sd':>11}"
          f"{'KASN mean':>13}{'KASN sd':>11}{'|D|/sd':>10}")
    print("-" * 92)
    for _, r in df.iterrows():
        print(f"{r['parameter']:<10}{r['truth']:>11.4f}{r['exact_mean']:>13.4f}"
              f"{r['exact_sd']:>11.4f}{r['kasn_mean']:>13.4f}"
              f"{r['kasn_sd']:>11.4f}{r['abs_diff_over_sd']:>10.3f}")
    print("=" * 92)


def _plot_posteriors(chain_e, chain_k, truth, param_names, out_path, extra=""):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = [n for n in param_names if n in chain_e or n.startswith('eta')]
    ncol = 3
    nrow = int(np.ceil(len(names) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 3.2 * nrow),
                             squeeze=False)
    for ax, name in zip(axes.ravel(), names):
        if name.startswith('eta'):
            k = int(name[-1]) - 1
            ve, vk, tv = chain_e['eta'][:, k], chain_k['eta'][:, k], truth['eta'][k]
        else:
            ve, vk, tv = chain_e[name], chain_k[name], truth[name]
        ax.hist(ve, bins=40, density=True, alpha=0.55, color='steelblue',
                label='exact $F$')
        ax.hist(vk, bins=40, density=True, alpha=0.55, color='darkorange',
                label='KASN $\\hat F$')
        ax.axvline(tv, color='red', lw=1.4, label='truth')
        ax.set_xlabel(name); ax.set_ylabel('density'); ax.legend(fontsize=7)
    for ax in axes.ravel()[len(names):]:
        ax.axis('off')
    fig.suptitle(f"Posterior comparison, DGP = {DGP} {extra}", fontsize=11)
    fig.tight_layout(); fig.savefig(out_path, dpi=140); plt.close(fig)


def save_chains(chain_e, chain_k, param_names):
    frames = []
    for tag, ch in (('exact', chain_e), ('kasn', chain_k)):
        n = len(ch['log_post'])
        keep = np.arange(0, n, max(1, CHAIN_THIN))
        d = {'oracle': tag, 'draw': keep, 'log_post': ch['log_post'][keep]}
        for name in param_names:
            if name.startswith('eta'):
                d[name] = ch['eta'][keep, int(name[-1]) - 1]
            elif name in ch:
                d[name] = ch[name][keep]
        frames.append(pd.DataFrame(d))
    df = pd.concat(frames, ignore_index=True)
    df['dgp'], df['seed'], df['run_id'] = DGP, SEED, RUN_ID
    path = _out_path('posterior_draws', 'csv')
    df.to_csv(path, index=False)
    return path


def print_config(cfg, dgp):
    print("\n" + "=" * 74)
    print("DDCM + KASN CONFIGURATION")
    print("=" * 74)
    print(f"  DGP:                {DGP}")
    print(f"  Features (d={dgp.input_dim}):    {dgp.feature_names}")
    print(f"  Mileage bins:       M = {dgp.M}")
    if DGP == 'ar1_gauss':
        print(f"  xi nodes:           K = {dgp.K} (Rouwenhorst)  "
              f"sigma_eta={dgp.sigma_eta}  sigma_nu={dgp.sigma_nu}")
        print(f"  Likelihood:         exact forward recursion, no augmentation")
    else:
        print(f"  Likelihood:         closed form (log-sum-exp)")
    print(f"  Design:             Beta({DESIGN_BETA_A}, {DESIGN_BETA_B}) on "
          f"continuous coordinates"
          + ("  [params uniform: (E.1) fails]" if DESIGN_UNIFORM_PARAMS else ""))
    print(f"  Training noise sd:  {TRAIN_NOISE_SD}"
          + ("   -> U_t v* correction active" if TRAIN_NOISE_SD > 0
             else "   -> noiseless, correction vanishes"))
    for k in ('n_train', 'n_val', 'n_test', 'depth', 'width', 'gamma', 'gl',
              'lr', 'batch_size', 'epochs', 'patience', 'n_buses', 'n_periods',
              'n_burn', 'n_iter'):
        print(f"    {k:<12s} = {cfg[k]}")
    print(f"  Pruning:            {PRUNING_THRESHOLD_METHOD}, prune-aware "
          f"selection = {PRUNE_AWARE_SELECTION}, fine-tune "
          f"{POST_PRUNE_FINETUNE_EPOCHS} epochs")
    print(f"  Average derivative: {COMPUTE_AVERAGE_DERIVATIVES}   "
          f"state={COMPUTE_STATE_DERIVATIVES}  param={COMPUTE_PARAM_DERIVATIVES}"
          f"  Riesz correction={AD_RIESZ_CORRECTION}")
    print(f"    coordinates:      {ad_labels_for(dgp)}")
    if COMPUTE_COUNTERFACTUAL:
        print(f"  Counterfactual:     {CF_PARAM} shrunk {CF_TAU:.0%} toward "
              f"{'the support midpoint' if CF_CENTER is None else CF_CENTER}")
    print(f"  Output:             {OUTPUT_DIR}   seed={SEED}   run_id={RUN_ID}")
    print(f"  Device:             {DEVICE}")
    print("=" * 74 + "\n")




MODE_PRESETS = {
    "quick": dict(n_train=600, n_val=150, n_test=250, depth=3, width=24,
                  gamma=0.30, gl=1e-3, lr=1e-3, batch_size=256, epochs=150,
                  patience=25, n_buses=40, n_periods=40, n_burn=400,
                  n_iter=1500, tuning_epochs=40, tuning_patience=8),
    "standard": dict(n_train=10000, n_val=2000, n_test=2000, depth=4, width=32,
                     gamma=0.30, gl=1e-3, lr=1e-3, batch_size=512, epochs=600,
                     patience=50, n_buses=70, n_periods=60, n_burn=2000,
                     n_iter=8000, tuning_epochs=60, tuning_patience=12),
}


def make_dgp():
    return (BusEngineEVIID(M=M_STATES) if DGP == 'ev_iid'
            else BusEngineAR1Gauss(M=M_STATES, K=XI_N_NODES,
                                   sigma_eta=SIGMA_ETA, sigma_nu=SIGMA_NU))


def make_truth() -> dict:
    t = dict(TRUTH_EV_IID if DGP == 'ev_iid' else TRUTH_AR1)
    t['eta'] = np.asarray(t['eta'], dtype=np.float64)
    return t



def design_coverage_diagnostic(chain, design, dgp) -> dict:
    out = {}
    for name in ('alpha1', 'alpha2', 'beta', 'rho'):
        if name not in chain or name not in design.spec:
            continue
        lo, hi, _ = design.spec[name]
        r = scipy.stats.beta.ppf([0.01, 0.99], design.a, design.b)
        c_lo, c_hi = lo + r[0] * (hi - lo), lo + r[1] * (hi - lo)
        v = np.asarray(chain[name], dtype=np.float64)
        out[f'outside_design_{name}'] = float(np.mean((v < c_lo) | (v > c_hi)))
        out[f'at_prior_edge_{name}'] = float(
            np.mean(np.isclose(v, design.prior_spec[name][0], atol=1e-6)
                    | np.isclose(v, design.prior_spec[name][1], atol=1e-6)))
    return out


def path_sup_error(dgp, model, chain, n_draws=200) -> dict:
    m = len(chain['log_post'])
    idx = np.unique(np.linspace(0, m - 1, min(n_draws, m)).astype(int))
    F_exact = make_F_oracle_exact(dgp)
    F_kasn = make_F_oracle_kasn(dgp, model)
    sup, l2 = [], []
    for i in idx:
        q = {k: float(chain[k][i]) for k in ('alpha1', 'alpha2', 'beta')
             if k in chain}
        if 'rho' in chain:
            q['rho'] = float(chain['rho'][i])
        q['eta'] = np.asarray(chain['eta'][i], dtype=np.float64)
        try:
            d = np.asarray(F_kasn(q)) - np.asarray(F_exact(q))
        except Exception:
            continue
        sup.append(float(np.max(np.abs(d))))
        l2.append(float(np.sqrt(np.mean(d ** 2))))
    if not sup:
        return {}
    return {'path_sup_error_mean': float(np.mean(sup)),
            'path_sup_error_max': float(np.max(sup)),
            'path_l2_error_mean': float(np.mean(l2)),
            'path_n_draws_checked': int(len(sup))}


def _eta_full_from_X(X, dgp):
    names = list(dgp.feature_names)
    e1 = np.asarray(X[:, names.index('eta1')], dtype=np.float64)
    e2 = np.asarray(X[:, names.index('eta2')], dtype=np.float64)
    return np.column_stack([e1, e2, 1.0 - e1 - e2])


def oracle_functionals_on_sample(dgp, design, X_eval, labels, verbose=True):
    t0 = time.time()
    eta_full = _eta_full_from_X(X_eval, dgp)
    names = list(dgp.feature_names)
    out = {}
    for lab in labels:
        j = names.index(lab)
        lo, hi, _ = design.spec[lab]
        h = TRUTH_FD_REL * (hi - lo)
        Xp, Xm = X_eval.copy(), X_eval.copy()
        Xp[:, j] = np.clip(X_eval[:, j] + h, lo, hi)
        Xm[:, j] = np.clip(X_eval[:, j] - h, lo, hi)
        step = Xp[:, j] - Xm[:, j]
        ok = step > 0
        d = (dgp.F_at(Xp[ok], eta_full[ok])
             - dgp.F_at(Xm[ok], eta_full[ok])) / step[ok]
        out[f'ad_{lab}_oracle'] = float(d.mean())
    if COMPUTE_COUNTERFACTUAL and CF_PARAM in names:
        j = names.index(CF_PARAM)
        lo, hi, _ = design.spec[CF_PARAM]
        c = 0.5 * (lo + hi) if CF_CENTER is None else float(CF_CENTER)
        Xc = X_eval.copy()
        Xc[:, j] = (1.0 - CF_TAU) * X_eval[:, j] + CF_TAU * c
        out['cf_oracle'] = float(np.mean(dgp.F_at(Xc, eta_full)
                                         - dgp.F_at(X_eval, eta_full)))
    out['_oracle_seconds'] = time.time() - t0
    if verbose:
        print(f"        oracle reference on {len(X_eval):,} test points in "
              f"{out['_oracle_seconds']:.1f}s")
    return out



def _in_notebook() -> bool:
    if any('ipykernel' in a or 'colab_kernel' in a for a in sys.argv):
        return True
    try:
        from IPython import get_ipython
        shell = get_ipython()
        return shell is not None and shell.__class__.__name__ != 'TerminalInteractiveShell'
    except Exception:
        return False


def _clean_argv(argv=None):
    if argv is not None:
        return list(argv)
    return [] if _in_notebook() else sys.argv[1:]


def get_seed() -> int:
    for a in _clean_argv():
        if not a.startswith('-'):
            try:
                return int(a)
            except ValueError:
                pass
    env_seed = os.getenv('SIMULATION_SEED')
    if env_seed is not None:
        try:
            return int(env_seed)
        except ValueError:
            pass
    task = os.getenv('SLURM_ARRAY_TASK_ID') or os.getenv('SLURM_PROCID')
    base_time = int(time.time() * 1000) % 2 ** 32
    seed = ((base_time + int(task) * 1_000_000) % 2 ** 32
            if task is not None else base_time)
    print(f"Generated seed: {seed} (task_id: {task})")
    return seed


def _slurm_context() -> dict:
    return {'hostname': socket.gethostname(),
            'slurm_job_id': os.getenv('SLURM_JOB_ID', ''),
            'slurm_proc_id': os.getenv('SLURM_PROCID', ''),
            'slurm_array_task_id': os.getenv('SLURM_ARRAY_TASK_ID', '')}



def _truth_key(dgp, design, labels) -> str:
    payload = {'dgp': dgp.name, 'M': dgp.M, 'labels': list(labels),
               'spec': {k: list(v) for k, v in design.spec.items()},
               'eta_conc': list(PRIOR_ETA_CONC),
               'a': design.a, 'b': design.b,
               'shapes': {k: list(v) for k, v in sorted(design.shapes.items())},
               'inflate': {k: DESIGN_INFLATE_BY_COORD.get(k, DESIGN_INFLATE)
                           for k in sorted(design.spec)},
               'cf': [CF_PARAM, CF_TAU, CF_CENTER],
               'n_truth': N_TRUTH, 'truth_seed': TRUTH_SEED,
               'fd_rel': TRUTH_FD_REL}
    if dgp.name == 'ar1_gauss':
        payload.update(K_nodes=dgp.K, s_eta=dgp.sigma_eta, s_nu=dgp.sigma_nu)
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def compute_true_functionals(dgp, design, labels, verbose=True) -> dict:
    t0 = time.time()
    rng = np.random.default_rng(TRUTH_SEED)
    X, eta_full = design.sample(N_TRUTH, rng)
    names = list(dgp.feature_names)
    out = {}
    F_base = dgp.F_at(X, eta_full)

    for lab in labels:
        j = names.index(lab)
        lo, hi, _ = design.spec[lab]
        h = TRUTH_FD_REL * (hi - lo)
        Xp, Xm = X.copy(), X.copy()
        Xp[:, j] = np.clip(X[:, j] + h, lo, hi)
        Xm[:, j] = np.clip(X[:, j] - h, lo, hi)
        step = Xp[:, j] - Xm[:, j]
        ok = step > 0
        d = (dgp.F_at(Xp[ok], eta_full[ok])
             - dgp.F_at(Xm[ok], eta_full[ok])) / step[ok]
        out[f'ad_{lab}_truth'] = float(d.mean())
        if verbose:
            print(f"        truth dF_0/d{lab:<7s} = {out[f'ad_{lab}_truth']:+.6f}"
                  f"   (n={int(ok.sum())}, h={h:.4g})")

    if COMPUTE_COUNTERFACTUAL and CF_PARAM in names:
        j = names.index(CF_PARAM)
        lo, hi, _ = design.spec[CF_PARAM]
        c = 0.5 * (lo + hi) if CF_CENTER is None else float(CF_CENTER)
        X_cf = X.copy()
        X_cf[:, j] = (1.0 - CF_TAU) * X[:, j] + CF_TAU * c
        out['cf_truth'] = float(np.mean(dgp.F_at(X_cf, eta_full) - F_base))
        if verbose:
            print(f"        truth counterfactual = {out['cf_truth']:+.6f}")

    out['_truth_seconds'] = time.time() - t0
    out['_n_truth'] = N_TRUTH
    return out


def merge_truth_parts(part_dir, results_dir, dgp, design, labels,
                      verbose=True) -> dict:
    import glob
    files = sorted(glob.glob(os.path.join(part_dir, f"truth_{dgp.name}_*.json")))
    if not files:
        raise ValueError(f"no truth parts found in {part_dir}")
    parts = [json.load(open(f)) for f in files]
    keys = [k for k in parts[0] if not k.startswith('_')]
    out = {k: float(np.mean([p[k] for p in parts])) for k in keys}
    out['_n_truth'] = int(sum(p.get('_n_truth', 0) for p in parts))
    out['_truth_seconds'] = -1.0
    if verbose:
        print(f"Merged {len(parts)} parts, {out['_n_truth']:,} points total")
        for k in keys:
            v = [p[k] for p in parts]
            print(f"  {k:<22s} {out[k]:+.6f}   sd across parts {np.std(v):.6f}"
                  f"   implied se of the merge {np.std(v)/np.sqrt(len(v)):.6f}")
    path = os.path.join(results_dir,
                        f"truth_{dgp.name}_{_truth_key(dgp, design, labels)}.json")
    fd, tmp = tempfile.mkstemp(dir=results_dir, suffix='.tmp')
    with os.fdopen(fd, 'w') as fh:
        json.dump(out, fh, indent=1)
    os.replace(tmp, path)
    if verbose:
        print(f"Wrote {path}")
    return out


def load_or_compute_truth(dgp, design, labels, results_dir, verbose=True) -> dict:
    path = os.path.join(results_dir,
                        f"truth_{dgp.name}_{_truth_key(dgp, design, labels)}.json")
    if USE_TRUTH_CACHE and os.path.exists(path):
        try:
            with open(path) as fh:
                truth = json.load(fh)
            if verbose:
                print(f"        truth from cache {os.path.basename(path)}")
            return truth
        except (json.JSONDecodeError, OSError):
            if verbose:
                print("        truth cache unreadable, recomputing")
    truth = compute_true_functionals(dgp, design, labels, verbose=verbose)
    if USE_TRUTH_CACHE:
        try:
            fd, tmp = tempfile.mkstemp(dir=results_dir, suffix='.tmp')
            with os.fdopen(fd, 'w') as fh:
                json.dump(truth, fh, indent=1)
            os.replace(tmp, path)
        except OSError as exc:
            print(f"        [warn] truth cache not written: {exc}")
    return truth



def run_replication(seed, dgp_name, mode, results_dir) -> pd.DataFrame:
    global DGP, SEED, OUTPUT_DIR
    DGP, SEED, OUTPUT_DIR = dgp_name, seed, results_dir
    torch.set_num_threads(max(1, THREADS))
    run_id = str(uuid.uuid4())[:8]

    cfg = dict(MODE_PRESETS[mode])
    if N_TRAIN_OVERRIDE is not None:
        cfg['n_train'] = int(N_TRAIN_OVERRIDE)
    if N_VAL_OVERRIDE is not None:
        cfg['n_val'] = int(N_VAL_OVERRIDE)
    if N_TEST_OVERRIDE is not None:
        cfg['n_test'] = int(N_TEST_OVERRIDE)
    dgp = make_dgp()
    truth_q = make_truth()
    labels = ad_labels_for(dgp)
    t_start = time.time()
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    if MC_VERBOSE:
        print(f"\n[rep] dgp={dgp_name} mode={mode} seed={seed} run_id={run_id} "
              f"d={dgp.input_dim} device={DEVICE}")

    design = DesignDistribution(dgp)
    truth_vals = (load_or_compute_truth(dgp, design, labels, results_dir,
                                        verbose=MC_VERBOSE)
                  if COMPUTE_TRUTH else {})

    panel = dgp.simulate_panel(truth_q, cfg["n_buses"], cfg["n_periods"], rng)
    n_repl = int((panel["d"] == 2).sum())
    data = build_training_data(dgp, design, cfg["n_train"], cfg["n_val"],
                               cfg["n_test"], rng, noise_sd=TRAIN_NOISE_SD,
                               verbose=MC_VERBOSE)

    gamma, lr, gl, width = cfg["gamma"], cfg["lr"], cfg["gl"], cfg["width"]
    sel_time = 0.0
    if SELECT_HP:
        gamma, lr, gl, width, sel_time, _ = select_kasn_hyperparams(
            data["X_train"], data["y_train"], data["X_val"], data["y_val"],
            n_samples=cfg["n_train"], input_dim=dgp.input_dim,
            depth=cfg["depth"], batch_size=cfg["batch_size"],
            tuning_epochs=cfg["tuning_epochs"],
            tuning_patience=cfg["tuning_patience"], verbose=MC_VERBOSE)

    t0 = time.time()
    model = KASN(input_dim=dgp.input_dim, n_samples=cfg["n_train"],
                 gamma=gamma, kasn_width=width, depth=cfg["depth"],
                 verbose=MC_VERBOSE)
    _, _, best_val_fit, _ = model.fit(
        data["X_train"], data["y_train"], data["X_val"], data["y_val"],
        epochs=cfg["epochs"], lr=lr, batch_size=cfg["batch_size"],
        patience=cfg["patience"], group_lasso_reg_scale=gl, verbose=MC_VERBOSE)
    val_pre, val_post, val_ft = prune_and_finetune(
        model, data["X_train"], data["y_train"], data["X_val"], data["y_val"],
        lr=lr, batch_size=cfg["batch_size"], group_lasso_reg_scale=gl,
        verbose=MC_VERBOSE)
    model.final_delta_projection(verbose=MC_VERBOSE)
    t_train = time.time() - t0

    pred = {s: model.predict(data[f"X_{s}"]) for s in ('train', 'val', 'test')}
    mets = {s: {'mse': float(np.mean((pred[s] - data[f"y_{s}"]) ** 2)),
                'rmse': float(np.sqrt(np.mean((pred[s] - data[f"y_{s}"]) ** 2))),
                'r2': _r2(data[f"y_{s}"], pred[s])} for s in pred}
    sp_rep = model.sparsity_report()
    pr_rep = model.penalty_rate_diagnostic()
    if MC_VERBOSE:
        print(f"[rep] KASN fit {t_train:.1f}s  test R2={mets['test']['r2']:.6f}")

    inf_boxes = infer_boxes(truth_q) if DECOUPLE_POSTERIOR_DESIGN else None
    if DECOUPLE_POSTERIOR_DESIGN:
        post_data = build_posterior_training_data(
            dgp, truth_q, inf_boxes, cfg["n_train"], cfg["n_val"],
            cfg["n_test"], rng, kind=POSTERIOR_DESIGN_KIND,
            noise_sd=TRAIN_NOISE_SD, verbose=MC_VERBOSE)
        t0 = time.time()
        model_post = KASN(input_dim=dgp.input_dim, n_samples=cfg["n_train"],
                          gamma=gamma, kasn_width=width, depth=cfg["depth"],
                          verbose=MC_VERBOSE)
        model_post.fit(
            post_data["X_train"], post_data["y_train"],
            post_data["X_val"], post_data["y_val"],
            epochs=cfg["epochs"], lr=lr, batch_size=cfg["batch_size"],
            patience=cfg["patience"], group_lasso_reg_scale=gl, verbose=MC_VERBOSE)
        prune_and_finetune(
            model_post, post_data["X_train"], post_data["y_train"],
            post_data["X_val"], post_data["y_val"], lr=lr,
            batch_size=cfg["batch_size"], group_lasso_reg_scale=gl,
            verbose=MC_VERBOSE)
        model_post.final_delta_projection(verbose=MC_VERBOSE)
        t_post = time.time() - t0
        post_test_r2 = _r2(post_data["y_test"],
                           model_post.predict(post_data["X_test"]))
        if MC_VERBOSE:
            print(f"[rep] posterior-F fit {t_post:.1f}s  "
                  f"test R2={post_test_r2:.6f}")
    else:
        model_post, post_test_r2, t_post = model, mets['test']['r2'], 0.0

    lik = (LikelihoodEVIID(panel, dgp.M) if dgp_name == 'ev_iid'
           else LikelihoodAR1(panel, dgp))
    prior = MCMCPrior(dgp, boxes=(inf_boxes if (DECOUPLE_POSTERIOR_DESIGN
                                  and MATCH_PRIOR_TO_INFER_BOX) else None))
    has_rho = (dgp_name == 'ar1_gauss')

    t0 = time.time()
    chain_e, diag_e = DDCMSampler(lik, prior, make_F_oracle_exact(dgp),
                                  has_rho).sample(cfg["n_iter"], cfg["n_burn"],
                                                  rng=rng)
    t_e = time.time() - t0
    t0 = time.time()
    chain_k, diag_k = DDCMSampler(lik, prior, make_F_oracle_kasn(dgp, model_post),
                                  has_rho).sample(cfg["n_iter"], cfg["n_burn"],
                                                  rng=rng)
    t_k = time.time() - t0
    if MC_VERBOSE:
        print(f"[rep] MCMC exact {t_e:.1f}s  KASN {t_k:.1f}s")

    ad = compute_average_derivatives(
        model, data["X_test"], data["X_train"], design, dgp,
        y_eval=data["y_test"], y_eval_pred=pred['test'], verbose=MC_VERBOSE)
    ccp = average_ccp_effect(model, dgp, data["X_test"], design,
                             verbose=MC_VERBOSE)
    cf = (counterfactual_difference(model, design, dgp, data["X_test"],
                                    y_eval=data["y_test"], verbose=MC_VERBOSE)
          if COMPUTE_COUNTERFACTUAL else {})

    row = {'seed': seed, 'run_id': run_id, 'dgp': dgp_name, 'mode': mode,
           'input_dim': dgp.input_dim, 'n_train': cfg['n_train'],
           'n_val': cfg['n_val'], 'n_test': cfg['n_test'],
           'n_buses': cfg['n_buses'], 'n_periods': cfg['n_periods'],
           'n_panel_obs': int(panel['x'].size), 'n_replacements': n_repl,
           'replacement_rate': n_repl / panel['x'].size,
           'n_iter': cfg['n_iter'], 'n_burn': cfg['n_burn'],
           'train_noise_sd': TRAIN_NOISE_SD, 'select_hp': SELECT_HP,
           'design_inflate': DESIGN_INFLATE,
           'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
           **_slurm_context()}
    row.update({f'{s}_{m}': mets[s][m] for s in mets for m in mets[s]})
    row.update({'kasn_gamma': gamma, 'kasn_lr': lr, 'kasn_gl': gl,
                'kasn_width': width, 'kasn_depth': cfg['depth'],
                'kasn_G_n': model.G, 'fit_seconds': t_train,
                'selection_seconds': sel_time, 'best_val_fit': best_val_fit,
                'val_loss_pre_prune': val_pre,
                'val_loss_post_prune': val_post,
                'val_loss_finetuned': val_ft})
    row.update(sp_rep)
    row.update({f'penalty_{k}': v for k, v in pr_rep.items()})
    row.update({'mcmc_exact_seconds': t_e, 'mcmc_kasn_seconds': t_k,
                'speedup': t_e / max(t_k, 1e-9),
                'exact_seconds_per_1000_draws':
                    1000.0 * t_e / max(cfg['n_iter'], 1),
                'kasn_seconds_per_1000_draws':
                    1000.0 * t_k / max(cfg['n_iter'], 1),
                'exact_mean_log_post': float(chain_e['log_post'].mean()),
                'kasn_mean_log_post': float(chain_k['log_post'].mean())})
    row.update({f'exact_{k}': v for k, v in diag_e.items() if k.startswith('acc_')})
    row.update({f'kasn_{k}': v for k, v in diag_k.items() if k.startswith('acc_')})

    summ = posterior_summary(chain_e, chain_k, truth_q, dgp.param_names)
    for _, r in summ.iterrows():
        p = r['parameter']
        row[f'post_{p}_truth'] = r['truth']
        row[f'post_{p}_exact_mean'] = r['exact_mean']
        row[f'post_{p}_exact_sd'] = r['exact_sd']
        row[f'post_{p}_kasn_mean'] = r['kasn_mean']
        row[f'post_{p}_kasn_sd'] = r['kasn_sd']
        row[f'post_{p}_absdiff_over_sd'] = r['abs_diff_over_sd']
        row[f'post_{p}_exact_cover95'] = int(
            r['exact_q025'] <= r['truth'] <= r['exact_q975'])
        row[f'post_{p}_kasn_cover95'] = int(
            r['kasn_q025'] <= r['truth'] <= r['kasn_q975'])

    for lab, info in ad.items():
        pre = f'ad_{lab}'
        for k_src, k_dst in (('coordinate_kind', 'kind'), ('mu_hat', 'mu_hat'),
                             ('se_hat', 'se_hat'), ('se_naive', 'se_naive'),
                             ('ci_lower', 'ci_lower'), ('ci_upper', 'ci_upper'),
                             ('riesz_gap', 'riesz_gap'),
                             ('boundary_condition_E1', 'E1'),
                             ('riesz_gap_abs', 'riesz_gap_abs'),
                             ('riesz_method', 'riesz_method'),
                             ('riesz_L2_finite', 'riesz_L2_finite'),
                             ('riesz_v_rms', 'riesz_v_rms'),
                             ('riesz_v_rms_theory', 'riesz_v_rms_theory'),
                             ('riesz_n_capped', 'riesz_n_capped'),
                             ('admissible', 'admissible')):
            row[f'{pre}_{k_dst}'] = info.get(k_src, np.nan)
        tv = truth_vals.get(f'{pre}_truth', np.nan)
        mu, se, sn = (row[f'{pre}_mu_hat'], row[f'{pre}_se_hat'],
                      row[f'{pre}_se_naive'])
        row[f'{pre}_truth'] = tv
        row[f'{pre}_ci_length'] = 2 * 1.96 * se if np.isfinite(se) else np.nan
        row[f'{pre}_error'] = mu - tv if np.isfinite(tv) else np.nan
        row[f'{pre}_covered'] = (int(abs(mu - tv) <= 1.96 * se)
                                 if np.isfinite(tv) and np.isfinite(se) else np.nan)
        row[f'{pre}_covered_naive'] = (int(abs(mu - tv) <= 1.96 * sn)
                                       if np.isfinite(tv) and np.isfinite(sn)
                                       else np.nan)

    if cf:
        row.update({f'cf_{k}': v for k, v in cf.items()})
        tv = truth_vals.get('cf_truth', np.nan)
        mu, se, sn = cf['gamma_cf'], cf['se_hat'], cf['se_naive']
        row['cf_truth'] = tv
        row['cf_ci_length'] = 2 * 1.96 * se
        row['cf_error'] = mu - tv if np.isfinite(tv) else np.nan
        row['cf_covered'] = (int(abs(mu - tv) <= 1.96 * se)
                             if np.isfinite(tv) else np.nan)
        row['cf_covered_naive'] = (int(abs(mu - tv) <= 1.96 * sn)
                                   if np.isfinite(tv) else np.nan)
    row['ccp_mean_effect'] = ccp.get('mean_ccp_effect', np.nan)
    row['riesz_method'] = AD_RIESZ_METHOD
    row['riesz_L2_finite'] = riesz_second_moment_finite(design)
    row['design_beta_a'] = design.a
    row['decouple_posterior'] = DECOUPLE_POSTERIOR_DESIGN
    row['posterior_design_kind'] = POSTERIOR_DESIGN_KIND if DECOUPLE_POSTERIOR_DESIGN else 'shared'
    row['match_prior_to_infer_box'] = (DECOUPLE_POSTERIOR_DESIGN
                                       and MATCH_PRIOR_TO_INFER_BOX)
    row['posterior_fit_seconds'] = t_post
    row['posterior_test_r2'] = post_test_r2
    if inf_boxes is not None:
        for k, (blo, bhi) in inf_boxes.items():
            row[f'infer_box_{k}_lo'] = blo
            row[f'infer_box_{k}_hi'] = bhi

    if COMPUTE_ORACLE_FUNCTIONALS:
        orc = oracle_functionals_on_sample(dgp, design, data["X_test"], labels,
                                           verbose=MC_VERBOSE)
        row['oracle_seconds'] = orc.pop('_oracle_seconds', np.nan)
        for lab in labels:
            pre = f'ad_{lab}'
            ov = orc.get(f'{pre}_oracle', np.nan)
            mu, se = row.get(f'{pre}_mu_hat', np.nan), row.get(f'{pre}_se_hat', np.nan)
            row[f'{pre}_oracle'] = ov
            row[f'{pre}_error_oracle'] = mu - ov if np.isfinite(ov) else np.nan
            row[f'{pre}_covered_oracle'] = (
                int(abs(mu - ov) <= 1.96 * se)
                if np.isfinite(ov) and np.isfinite(se) else np.nan)
            row[f'{pre}_oracle_minus_truth'] = (
                ov - row.get(f'{pre}_truth', np.nan) if np.isfinite(ov) else np.nan)
        if cf:
            ov = orc.get('cf_oracle', np.nan)
            mu, se = cf['gamma_cf'], cf['se_hat']
            row['cf_oracle'] = ov
            row['cf_error_oracle'] = mu - ov if np.isfinite(ov) else np.nan
            row['cf_covered_oracle'] = (int(abs(mu - ov) <= 1.96 * se)
                                        if np.isfinite(ov) else np.nan)
            row['cf_oracle_minus_truth'] = (
                ov - row.get('cf_truth', np.nan) if np.isfinite(ov) else np.nan)

    if RUN_FFANN:
        t0 = time.time()
        ff = FFANNNorets(input_dim=dgp.input_dim, verbose=MC_VERBOSE)
        ff_val = ff.fit(data["X_train"], data["y_train"],
                        data["X_val"], data["y_val"], verbose=MC_VERBOSE)
        t_ff = time.time() - t0
        ff_pred = {sp: ff.predict(data[f"X_{sp}"]) for sp in ('train', 'val', 'test')}
        row.update({f'ffann_{sp}_r2': _r2(data[f"y_{sp}"], ff_pred[sp])
                    for sp in ff_pred})
        row.update({'ffann_test_rmse':
                        float(np.sqrt(np.mean((ff_pred['test']
                                               - data["y_test"]) ** 2))),
                    'ffann_fit_seconds': t_ff, 'ffann_best_val': ff_val,
                    'ffann_hidden': FFANN_HIDDEN})
        if MC_VERBOSE:
            print(f"[rep] FFANN fit {t_ff:.1f}s  test R2="
                  f"{row['ffann_test_r2']:.6f}")

        ff_ad = compute_average_derivatives(
            ff, data["X_test"], data["X_train"], design, dgp,
            y_eval=data["y_test"], y_eval_pred=ff_pred['test'],
            verbose=False)
        for lab, info in ff_ad.items():
            pre = f'ffann_ad_{lab}'
            mu, se = info.get('mu_hat', np.nan), info.get('se_hat', np.nan)
            row[f'{pre}_mu_hat'], row[f'{pre}_se_hat'] = mu, se
            tv = row.get(f'ad_{lab}_truth', np.nan)
            row[f'{pre}_error'] = mu - tv if np.isfinite(tv) else np.nan
            row[f'{pre}_covered'] = (int(abs(mu - tv) <= 1.96 * se)
                                     if np.isfinite(tv) and np.isfinite(se)
                                     else np.nan)
            ov = row.get(f'ad_{lab}_oracle', np.nan)
            row[f'{pre}_covered_oracle'] = (int(abs(mu - ov) <= 1.96 * se)
                                            if np.isfinite(ov) and np.isfinite(se)
                                            else np.nan)
        if COMPUTE_COUNTERFACTUAL:
            ff_cf = counterfactual_difference(ff, design, dgp, data["X_test"],
                                              y_eval=data["y_test"], verbose=False)
            if ff_cf:
                mu, se = ff_cf['gamma_cf'], ff_cf['se_hat']
                row['ffann_cf_gamma_cf'], row['ffann_cf_se_hat'] = mu, se
                tv = row.get('cf_truth', np.nan)
                row['ffann_cf_error'] = mu - tv if np.isfinite(tv) else np.nan
                row['ffann_cf_covered'] = (int(abs(mu - tv) <= 1.96 * se)
                                           if np.isfinite(tv) else np.nan)

        if FFANN_MCMC:
            if DECOUPLE_POSTERIOR_DESIGN:
                ff_post = FFANNNorets(input_dim=dgp.input_dim, verbose=MC_VERBOSE)
                ff_post.fit(post_data["X_train"], post_data["y_train"],
                            post_data["X_val"], post_data["y_val"],
                            verbose=MC_VERBOSE)
            else:
                ff_post = ff
            t0 = time.time()
            chain_f, diag_f = DDCMSampler(lik, prior,
                                          make_F_oracle_kasn(dgp, ff_post),
                                          has_rho).sample(cfg["n_iter"],
                                                          cfg["n_burn"], rng=rng)
            row['mcmc_ffann_seconds'] = time.time() - t0
            row.update({f'ffann_{k}': v for k, v in diag_f.items()
                        if k.startswith('acc_')})
            summ_f = posterior_summary(chain_e, chain_f, truth_q, dgp.param_names)
            for _, r in summ_f.iterrows():
                pp = r['parameter']
                row[f'ffann_post_{pp}_mean'] = r['kasn_mean']
                row[f'ffann_post_{pp}_sd'] = r['kasn_sd']
                row[f'ffann_post_{pp}_absdiff_over_sd'] = r['abs_diff_over_sd']
                row[f'ffann_post_{pp}_cover95'] = int(
                    r['kasn_q025'] <= r['truth'] <= r['kasn_q975'])

    if COMPUTE_PATH_DIAGNOSTICS:
        row.update(design_coverage_diagnostic(chain_k, design, dgp))
        pe = path_sup_error(dgp, model_post, chain_k, PATH_N_DRAWS)
        row.update(pe)
        if pe:
            nT = float(panel['x'].size)
            row['nT_times_path_sup_error'] = nT * pe['path_sup_error_mean']
            if MC_VERBOSE:
                print(f"[rep] path sup error {pe['path_sup_error_mean']:.4f}  "
                      f"nT x sup = {row['nT_times_path_sup_error']:.1f}")

    row['truth_seconds'] = truth_vals.get('_truth_seconds', np.nan)
    row['n_truth'] = truth_vals.get('_n_truth', np.nan)
    row['total_seconds'] = time.time() - t_start
    return pd.DataFrame([row])



def _run_self_check() -> None:
    print("=" * 74)
    print("SELF-CHECK")
    print("=" * 74)
    rng = np.random.default_rng(0)
    for dgp in (BusEngineEVIID(M=M_STATES),
                BusEngineAR1Gauss(M=M_STATES, K=XI_N_NODES)):
        des = DesignDistribution(dgp)
        for name, (lo, hi, _) in des.spec.items():
            plo, phi = des.prior_spec[name]
            assert lo <= plo + 1e-9 and hi >= phi - 1e-9, (
                f"{dgp.name}/{name}: design [{lo},{hi}] must contain the prior "
                f"[{plo},{phi}]")
        w_end = des.weight_on_ranks(np.array([1e-9, 1 - 1e-9]), coord=0)
        assert np.all(w_end < 1e-3), "design weight must vanish at the boundary"
        fails = [des.names[j] for j in range(dgp.input_dim)
                 if not des.satisfies_E1(j)]
        print(f"[{dgp.name}] design contains the prior on every coordinate; "
              f"(E.1) fails for {fails or 'none'}")

        src = TRUTH_EV_IID if dgp.name == 'ev_iid' else TRUTH_AR1
        q = dict(src); q['eta'] = np.asarray(q['eta'], dtype=np.float64)
        p = dgp.simulate_panel(q, 40, 40, np.random.default_rng(1))
        print(f"[{dgp.name}] panel {p['x'].shape}, replace freq "
              f"{(p['d'] == 2).mean():.4f}")
        X, ef = des.sample(6, rng)
        assert np.isfinite(dgp.F_at(X, ef)).all(), "F_at must be finite"
    print("Self-check completed successfully.")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="KASN DDCM Monte Carlo replication (SLURM ready)")
    parser.add_argument("seed", nargs="?", type=int, default=None)
    parser.add_argument("--dgp", choices=sorted(_VALID_DGPS), default=None)
    parser.add_argument("--mode", choices=["quick", "standard"], default=None)
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument("--noise-sd", type=float, default=None)
    parser.add_argument("--n-train", type=int, default=None,
                        help="override the training size N (for the N sweep); "
                             "validation and test sets stay at the preset size")
    parser.add_argument("--n-val", type=int, default=None)
    parser.add_argument("--n-test", type=int, default=None)
    parser.add_argument("--select-hp", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--merge-truth", type=str, default=None,
                        metavar="PART_DIR",
                        help="average truth_*.json files from PART_DIR into "
                             "the cache slot this configuration will read")
    parser.add_argument("--warm-truth-only", action="store_true",
                        help="compute and cache the population values, then "
                             "exit; run once before the task fan-out")
    args, _ = parser.parse_known_args(_clean_argv(argv))

    global DGP, TRAIN_NOISE_SD, SELECT_HP, OUTPUT_DIR
    global N_TRAIN_OVERRIDE, N_VAL_OVERRIDE, N_TEST_OVERRIDE
    dgp_name = args.dgp or DGP
    mode = args.mode or MODE_MC
    results_dir = args.outdir or OUTPUT_DIR
    if args.noise_sd is not None:
        TRAIN_NOISE_SD = args.noise_sd
    if args.n_train is not None:
        N_TRAIN_OVERRIDE = args.n_train
    if args.n_val is not None:
        N_VAL_OVERRIDE = args.n_val
    if args.n_test is not None:
        N_TEST_OVERRIDE = args.n_test
    if args.select_hp:
        SELECT_HP = True
    DGP, OUTPUT_DIR = dgp_name, results_dir
    os.makedirs(results_dir, exist_ok=True)

    if args.self_check:
        _run_self_check()
        return

    if args.merge_truth:
        dgp = make_dgp()
        design = DesignDistribution(dgp)
        merge_truth_parts(args.merge_truth, results_dir, dgp, design,
                          ad_labels_for(dgp))
        return

    if args.warm_truth_only:
        dgp = make_dgp()
        design = DesignDistribution(dgp)
        labels = ad_labels_for(dgp)
        print(f"Warming the truth cache: dgp={dgp_name}, labels={labels}")
        t = load_or_compute_truth(dgp, design, labels, results_dir)
        print({k: v for k, v in t.items() if not k.startswith('_')})
        return

    seed = args.seed if args.seed is not None else get_seed()
    np.random.seed(seed)
    if _in_notebook():
        print("[notebook] command-line arguments ignored; using the "
              "CONFIGURATION block above. Call main(['--self-check']) or "
              "main(['--mode', 'quick']) to override.")
    print("=" * 74)
    print(f"KASN DDCM MONTE CARLO REPLICATION  dgp={dgp_name}  mode={mode}")
    print(f"  seed={seed}  host={socket.gethostname()}  "
          f"job={os.getenv('SLURM_JOB_ID', '-')}  "
          f"proc={os.getenv('SLURM_PROCID', '-')}")
    print("=" * 74)

    df = run_replication(seed, dgp_name, mode, results_dir)
    run_id = df.iloc[0]['run_id']
    path = os.path.join(results_dir,
                        f"ddcm_mc_results_{dgp_name}_seed{seed}_{run_id}.csv")
    df.to_csv(path, index=False)
    print(f"\nWrote {path}")
    print(f"  test R2       {df.iloc[0]['test_r2']:.6f}")
    for c in sorted(c for c in df.columns if c.endswith('_covered')):
        print(f"  {c:<26s} {df.iloc[0][c]}")
    print(f"  total_seconds {df.iloc[0]['total_seconds']:.1f}")


if __name__ == "__main__":
    main()