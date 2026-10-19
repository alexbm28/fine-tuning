# CELL 1 -- COMPUTATIONS.
# Run once; rerun only to change the experiment or Lanczos iteration count.

from pathlib import Path
import json, math, time
import numpy as np
import pandas as pd
from scipy.linalg import svd, eigh_tridiagonal
from threadpoolctl import threadpool_limits

# -------------------- EXPERIMENT SETTINGS --------------------
SEED = 2026
D = 400
PAIRS = [(80, 320), (80, 80), (320, 80)]   # Left -> right.
BASE_N = 80
BUDGET_UNITS = 65536                     # Same raw B/(80*d) in every panel.
GAMMA_P = GAMMA_F = 0.05
ETA_P = ETA_F = 0.05
NOISE_SD = 0.10
ROTATE_DOWNSTREAM = True

LANCZOS_STEPS = 40                       # Fixed before reference calculations.
MAX_SWEEP_POINTS = 300_000
DATA_DIR = Path("neurips_figure1_lanczos_data")
# -------------------------------------------------------------

if not all(1 < n < D and 1 < nf < D for n, nf in PAIRS):
    raise ValueError("This experiment requires 1 < n,n' < d.")
if BUDGET_UNITS <= 0 or int(BUDGET_UNITS) != BUDGET_UNITS:
    raise ValueError("BUDGET_UNITS must be a positive integer.")
if min(GAMMA_P, GAMMA_F, ETA_P, ETA_F) <= 0 or NOISE_SD < 0:
    raise ValueError("Penalties and step sizes must be positive; noise >= 0.")
if int(LANCZOS_STEPS) != LANCZOS_STEPS or LANCZOS_STEPS < 1:
    raise ValueError("LANCZOS_STEPS must be a positive integer.")

DATA_DIR.mkdir(parents=True, exist_ok=True)
B = int(BUDGET_UNITS) * BASE_N * D


def make_data():
    # Identical RNG sequence to the previous sample-size experiment.
    rng = np.random.default_rng(SEED)
    Qp = np.linalg.qr(rng.normal(size=(D, D)))[0]
    Qf = (
        np.linalg.qr(rng.normal(size=(D, D)))[0]
        if ROTATE_DOWNSTREAM else Qp
    )
    spectrum = np.geomspace(1.0, 1e-3, D)

    X = (
        rng.normal(size=(max(n for n, _ in PAIRS), D))
        * np.sqrt(spectrum)
    ) @ Qp.T
    Xf = (
        rng.normal(size=(max(nf for _, nf in PAIRS), D))
        * np.sqrt(spectrum)
    ) @ Qf.T

    teacher = rng.normal(size=D) / np.sqrt(D)
    shift = rng.normal(size=D) / np.sqrt(D)
    y = X @ teacher + NOISE_SD * rng.normal(size=len(X))
    yf = Xf @ (teacher + shift) + NOISE_SD * rng.normal(size=len(Xf))
    return X, y, Xf, yf


def lanczos_upstream(X, y, gamma, steps):
    """Matrix-free Lanczos on H, starting from g = X.T @ y / n.

    No ridge solution, SVD or reference eigenvalue is used here.

    Write each Lanczos vector as q = X.T @ v and carry v as well as q.
    This implements the same recurrence while preventing accumulated
    roundoff leakage into the inactive parameter-space nullspace.
    Two-pass reorthogonalisation uses both representations.
    """
    n, d = X.shape
    if (
        y.shape != (n,)
        or not np.all(np.isfinite(X))
        or not np.all(np.isfinite(y))
    ):
        raise ValueError("Invalid or non-finite upstream data.")

    m = min(int(steps), n, d)
    g = X.T @ y / n
    gnorm = np.linalg.norm(g)
    if gnorm == 0:
        raise ValueError(
            "Zero initial gradient: the interior case does not apply."
        )

    q, v = g / gnorm, y / (n * gnorm)
    vprev, beta_prev = np.zeros(n), 0.0
    Q, V = np.empty((d, m)), np.empty((n, m))
    diagonal, offdiagonal, history = [], [], []
    started = time.perf_counter()

    for j in range(m):
        Q[:, j], V[:, j] = q, v
        Xq = X @ q
        alpha = float((Xq @ Xq) / n + gamma * (q @ q))

        # Hq - alpha*q - beta_prev*qprev, represented as X.T @ z.
        z = Xq / n + (gamma - alpha) * v - beta_prev * vprev
        r = X.T @ z

        for _ in range(2):
            z -= V[:, :j+1] @ (Q[:, :j+1].T @ r)
            r = X.T @ z

        beta = float(np.linalg.norm(r))
        diagonal.append(alpha)

        # Only a small (j+1)-dimensional tridiagonal problem is diagonalised.
        if j == 0:
            estimate, residual = alpha, beta
        else:
            ev, vectors = eigh_tridiagonal(
                np.array(diagonal),
                np.array(offdiagonal),
                select="i",
                select_range=(0, 0)
            )
            estimate = float(ev[0])
            residual = float(beta * abs(vectors[-1, 0]))

        history.append(dict(
            iteration=j+1,
            lambda_estimate=estimate,
            ritz_residual_indicator=residual,
            design_matvecs=1+4*(j+1),
            elapsed_seconds=time.perf_counter()-started
        ))

        if not np.isfinite(estimate) or estimate < gamma * (1-1e-10):
            raise FloatingPointError("Unresolved Lanczos estimate.")

        # Stop only at the predeclared limit or numerical breakdown.
        # No reference eigenvalue or allocation error determines stopping.
        if beta <= 1e-13 * max(1.0, abs(alpha), beta_prev) or j+1 == m:
            break

        offdiagonal.append(beta)
        vprev, v, q = v, z / beta, r / beta
        beta_prev = beta

    return estimate, history


def ridge_reference(X, y, gamma):
    """Offline spectral reference; never called by Lanczos."""
    U, s, Vh = svd(X / np.sqrt(len(y)), full_matrices=False)
    U, s, V = U[:, ::-1], s[::-1], Vh[::-1].T

    if s[0] <= 1e-10 * s[-1]:
        raise ValueError("The design is not numerically full row rank.")

    covariance = s**2
    h = covariance + gamma
    coefficients = s * (U.T @ (y / np.sqrt(len(y)))) / h
    return covariance, h, V, coefficients


def compute_panel(X, y, Xf, yf, estimate, history):
    n, nf = len(y), len(yf)
    cp, cf = n * D, nf * D                   # Declared cost proxies.
    L = math.lcm(cp, cf)

    if B % L:
        raise ValueError(f"Budget is incompatible with (n,n')={(n,nf)}.")

    T, rp, rf = B // L, L // cp, L // cf
    if T + 1 > MAX_SWEEP_POINTS:
        raise ValueError(
            "Too many allocations: reduce the budget "
            "or raise MAX_SWEEP_POINTS."
        )

    # Freeze the practical prediction BEFORE either ridge solve.
    # Here n' < d and independent Gaussian designs make the downstream
    # nullspace active almost surely for H'-norm: lambda_f = gamma_f.
    if not (0 < ETA_P*estimate < 1 and 0 < ETA_F*GAMMA_F < 1):
        raise ValueError("Estimated contraction factor is outside (0,1).")

    mu_p_est = rp * (-np.log1p(-ETA_P * estimate))
    mu_f_known = rf * (-np.log1p(-ETA_F * GAMMA_F))
    theta_est = float(mu_f_known / (mu_p_est + mu_f_known))

    # Everything below is OFFLINE REFERENCE work, not estimator input.
    sp, hp, Vp, ap = ridge_reference(X, y, GAMMA_P)
    sf, hf, Vf, af = ridge_reference(Xf, yf, GAMMA_F)

    if ETA_P * hp[-1] >= 1 or ETA_F * hf[-1] >= 1:
        raise ValueError(
            "Reduce the learning rates: the non-oscillatory bound fails."
        )
    if abs(ap[0]) <= 1e-12 * np.linalg.norm(ap):
        raise ValueError(
            "The slowest upstream component is numerically unresolved."
        )

    w, wf = Vp @ ap, Vf @ af
    Delta = wf - w
    cross = Vf.T @ Vp
    nullmap = Vp - Vf @ cross
    df, d0 = af - cross @ ap, -nullmap @ ap

    if np.linalg.norm(d0) <= 1e-10 * np.linalg.norm(w):
        raise ValueError(
            "The downstream nullspace is not resolved as active."
        )

    ratep = -np.log1p(-ETA_P * hp)
    ratef = -np.log1p(-ETA_F * hf)
    rate0f = -np.log1p(-ETA_F * GAMMA_F)
    mu_p, mu_f = rp * ratep[0], rf * rate0f
    theta = mu_f / (mu_p + mu_f)

    # Exact spectral information for the two-term reference prediction.
    u = Vp[:, 0] * ap[0]
    a = float(np.sum(sf * (Vf.T @ u)**2) + GAMMA_F * (u @ u))
    b = float(GAMMA_F * (d0 @ d0))
    d_cross = float(GAMMA_F * (u @ d0))
    z = (mu_f - mu_p) * d_cross

    root = np.hypot(z, 2 * np.sqrt(mu_p * mu_f * a * b))
    logq = (
        np.log(root + z) - np.log(2 * mu_f * b)
        if z >= 0
        else np.log(2 * mu_p * a) - np.log(root - z)
    )
    offset = logq / (mu_p + mu_f)

    def log_objective(k, ell):
        # Rescaled exact GD residuals avoid underflow.
        k, ell = np.broadcast_arrays(np.atleast_1d(k), np.atleast_1d(ell))
        out = np.empty(k.size)
        k, ell = k.reshape(-1), ell.reshape(-1)

        for start in range(0, len(k), 512):
            sl = slice(start, start + 512)
            kp, lf = k[sl, None], ell[sl, None]
            scale = np.minimum(kp * ratep[0], lf * rate0f)
            p = ap * np.exp(-kp * ratep + scale)

            row = (
                (-np.expm1(-lf * ratef)) * (p @ cross.T)
                - df * np.exp(-lf * ratef + scale)
            )
            null = (
                (-np.expm1(-lf * rate0f)) * (p @ nullmap.T)
                - d0 * np.exp(-lf * rate0f + scale)
            )

            norm2 = (
                np.sum(hf * row**2, axis=1)
                + GAMMA_F * np.sum(null**2, axis=1)
            )
            if np.any(norm2 <= 0) or not np.all(np.isfinite(norm2)):
                raise FloatingPointError(
                    "Unresolved objective; do not plot these results."
                )

            out[sl] = np.log(norm2) - 2 * scale[:, 0]

        return out

    # Independent check against direct GD at a small budget.
    for q in (0, 3, 6, 12):
        k, ell = rp * q, rf * (12 - q)
        wp = np.zeros(D)

        for _ in range(k):
            wp -= ETA_P * (
                X.T @ (X @ wp - y) / n + GAMMA_P * wp
            )

        wend = wp.copy()
        for _ in range(ell):
            wend -= ETA_F * (
                Xf.T @ (Xf @ wend - yf) / nf + GAMMA_F * wend
            )

        error = wend - wp - Delta
        direct = (
            np.linalg.norm(Xf @ error)**2 / nf
            + GAMMA_F * (error @ error)
        )
        np.testing.assert_allclose(
            np.exp(log_objective(k, ell)[0]),
            direct, rtol=1e-9, atol=1e-12
        )

    # Evaluate every feasible allocation for the numerical reference.
    t = np.arange(T + 1, dtype=np.int64)
    logf = log_objective(rp * t, rf * (T - t))
    tstar = int(np.argmin(logf))
    t0 = float(theta * T)
    centre = float(t0 + offset)

    t_est = float(theta_est * T)
    t_est_discrete = int(np.clip(np.floor(t_est + 0.5), 0, T))

    meta = dict(
        n=n, n_prime=nf, B=B, T=T, c_p=cp, c_f=cf, L=L,
        t_opt=tstar,
        t_leading_exact=t0,
        t_lanczos=t_est,
        t_lanczos_discrete=t_est_discrete,
        t_two_term=centre,
        offset=float(offset),
        leading_exact_percent=100*t0/T,
        lanczos_percent=100*theta_est,
        two_term_percent=100*centre/T,
        optimum_percent=100*tstar/T,
        k_star=rp*tstar,
        ell_star=rf*(T-tstar),
        k_lanczos=rp*t_est_discrete,
        ell_lanczos=rf*(T-t_est_discrete),
        lambda_p_reference=float(hp[0]),
        lambda_p_estimate=float(estimate),
        lambda_f_used=float(GAMMA_F),
        lanczos_iterations=len(history),
        lanczos_design_matvecs=history[-1]["design_matvecs"],
        lanczos_seconds=history[-1]["elapsed_seconds"],
        leading_estimation_error_percentage_points=100*abs(theta_est-theta),
        two_term_error_steps=abs(tstar-centre),
        lanczos_allocation_error_steps=abs(tstar-t_est),
        lanczos_log10_objective_ratio=float(
            (logf[t_est_discrete]-logf[tstar])/np.log(10)
        )
    )
    return meta, logf / np.log(10.0)


metadata, arrays, histories = [], {}, []

with threadpool_limits(limits=2):
    X, y, Xf, yf = make_data()

    # Complete ALL estimation runs before ANY SVD/reference calculation.
    # The n=80 upstream data are shared by two panels: reuse that estimate.
    estimates = {}
    for n in sorted({n for n, _ in PAIRS}):
        estimate, history = lanczos_upstream(
            X[:n], y[:n], GAMMA_P, LANCZOS_STEPS
        )
        estimates[n] = (estimate, history)
        histories.extend(dict(n=n, **item) for item in history)
        print(
            f"Lanczos n={n}: {len(history)} iterations; "
            f"estimate={estimate:.8f}"
        )

    for i, (n, nf) in enumerate(PAIRS):
        estimate, history = estimates[n]
        meta, log10f = compute_panel(
            X[:n], y[:n], Xf[:nf], yf[:nf], estimate, history
        )
        metadata.append(meta)
        arrays[f"log10f_{i}"] = log10f
        print(
            f"Reference n={n}, n'={nf}: "
            f"{meta['T'] + 1:,} allocations checked."
        )

np.savez_compressed(DATA_DIR / "curves.npz", **arrays)

configuration = dict(
    schema="lanczos_figure1_v1",
    seed=SEED, d=D, pairs=PAIRS, base_n=BASE_N,
    budget_units=int(BUDGET_UNITS), B=B,
    gamma_p=GAMMA_P, gamma_f=GAMMA_F,
    eta_p=ETA_P, eta_f=ETA_F,
    noise_sd=NOISE_SD,
    rotated_downstream=ROTATE_DOWNSTREAM,
    lanczos_steps=LANCZOS_STEPS,
    objective="squared Hprime norm",
    panels=metadata,
    estimator="Matrix-free Lanczos on H, starting from X.T@y/n",
    downstream_method="Known ridge nullspace eigenvalue gamma_f for this design",
    stopping="Fixed iteration limit or numerical breakdown; no reference input",
    budget_note="B is the training budget; estimation overhead is not deducted"
)
(DATA_DIR / "metadata.json").write_text(
    json.dumps(configuration, indent=2)
)

summary = pd.DataFrame(metadata)
summary.to_csv(DATA_DIR / "summary.csv", index=False)
pd.DataFrame(histories).to_csv(
    DATA_DIR / "lanczos_history.csv", index=False
)

print("\nDirect-GD checks passed. All estimates were fixed before reference SVDs.")
print(
    summary[[
        "n", "n_prime", "lanczos_percent", "leading_exact_percent",
        "two_term_percent", "optimum_percent", "lanczos_iterations"
    ]].round(4).to_string(index=False)
)
print("Percentages are compute allocated to pretraining.")
print("B excludes estimation overhead; this figure is not a timing comparison.")
print(f"Data saved in {DATA_DIR}. Rerun only Cell 2 to restyle the figure.")