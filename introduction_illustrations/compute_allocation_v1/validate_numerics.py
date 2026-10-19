#!/usr/bin/env python3
"""Validate the supplied experiment without repeating its allocation sweep.

This script extracts the supplied Cell 1 generator and numerical utilities by
AST. It preserves their RNG ordering and thin-SVD representation, and never
executes the cell's top-level experiment or replaces its saved objective arrays.
The independent long-double check recombines a full residual vector and applies
the downstream design directly, rather than reusing the original norm formula.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
import platform
from pathlib import Path

import numpy as np
import scipy
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "source" / "original_compute_cell.py"
CACHE = ROOT / "neurips_figure1_lanczos_data"
SELECTED = (0, 2)


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array_record(value):
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "sha256_c_order_bytes": hashlib.sha256(value.tobytes(order="C")).hexdigest(),
    }


def load_original_utilities():
    """Load only imports, experiment constants, and function definitions."""
    tree = ast.parse(SOURCE.read_text())
    settings = {
        "SEED", "D", "PAIRS", "BASE_N", "BUDGET_UNITS", "GAMMA_P", "GAMMA_F",
        "ETA_P", "ETA_F", "NOISE_SD", "ROTATE_DOWNSTREAM", "LANCZOS_STEPS",
        "MAX_SWEEP_POINTS", "DATA_DIR", "B",
    }
    keep = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef)):
            keep.append(copy.deepcopy(node))
        elif isinstance(node, ast.Assign):
            names = {target.id for target in node.targets if isinstance(target, ast.Name)}
            if names and names <= settings:
                keep.append(copy.deepcopy(node))

    # Reuse the exact original residual evaluator, stopping immediately after
    # its definition. This avoids the original all-allocation sweep and direct
    # GD loop. All preceding reference construction and guards are preserved.
    panel = copy.deepcopy(next(node for node in tree.body
                               if isinstance(node, ast.FunctionDef)
                               and node.name == "compute_panel"))
    panel.name = "prepare_original_evaluator"
    stop = next(i for i, node in enumerate(panel.body)
                if isinstance(node, ast.FunctionDef) and node.name == "log_objective")
    panel.body = panel.body[:stop + 1]
    panel.body.append(ast.Return(value=ast.Call(
        func=ast.Name(id="locals", ctx=ast.Load()), args=[], keywords=[])))
    keep.append(panel)
    module = ast.fix_missing_locations(ast.Module(body=keep, type_ignores=[]))
    namespace = {"__file__": str(SOURCE), "__name__": "original_cell_utilities"}
    exec(compile(module, str(SOURCE), "exec"), namespace)
    return namespace


def hprime_norm_squared(error, Xf, gamma):
    return np.sum((Xf @ error) ** 2) / len(Xf) + gamma * np.sum(error ** 2)


def independent_longdouble_log10(k, ell, state, Xf, gamma):
    """Rescaled residual with direct H'-norm, using long-double operations.

    SVD factors remain the supplied experiment's double-precision factors.
    Upstream C^k w_hat is always formed in the upstream thin-SVD span.
    No full upstream eigendecomposition or numerical nullspace is introduced.
    """
    ld = np.longdouble
    Vp = np.asarray(state["Vp"], dtype=ld)
    Vf = np.asarray(state["Vf"], dtype=ld)
    ap = np.asarray(state["ap"], dtype=ld)
    af = np.asarray(state["af"], dtype=ld)
    hp = np.asarray(state["hp"], dtype=ld)
    hf = np.asarray(state["hf"], dtype=ld)
    eta_p, eta_f = ld(state["eta_p_check"]), ld(state["eta_f_check"])
    gamma = ld(gamma)
    ratep = -np.log1p(-eta_p * hp)
    ratef = -np.log1p(-eta_f * hf)
    rate0 = -np.log1p(-eta_f * gamma)
    scale = min(ld(k) * ratep[0], ld(ell) * rate0)
    upstream_scaled = Vp @ (ap * np.exp(-ld(k) * ratep + scale))
    w = Vp @ ap
    projected_upstream = Vf.T @ upstream_scaled
    projected_w = Vf.T @ w
    upstream_null = upstream_scaled - Vf @ projected_upstream
    w_null = w - Vf @ projected_w

    # [I - C'^ell] C^k w_hat, followed by -C'^ell Delta, with
    # Delta's null component exactly -P'_0 w_hat by model construction.
    residual = Vf @ (-np.expm1(-ld(ell) * ratef) * projected_upstream)
    residual += -np.expm1(-ld(ell) * rate0) * upstream_null
    residual -= Vf @ (np.exp(-ld(ell) * ratef + scale) * (af - projected_w))
    residual += np.exp(-ld(ell) * rate0 + scale) * w_null
    norm2 = hprime_norm_squared(residual, np.asarray(Xf, dtype=ld), gamma)
    assert norm2 > 0 and np.isfinite(norm2)
    return float((np.log(norm2) - 2 * scale) / np.log(ld(10)))


def validate_selected(index, meta, saved, original, data):
    Xall, yall, Xfall, yfall = data
    n, nf = meta["n"], meta["n_prime"]
    X, y, Xf, yf = Xall[:n], yall[:n], Xfall[:nf], yfall[:nf]
    state = original["prepare_original_evaluator"](
        X, y, Xf, yf, meta["lambda_p_estimate"], [])
    stable = state["log_objective"]
    state.update(eta_p_check=original["ETA_P"], eta_f_check=original["ETA_F"])
    gamma_p, gamma_f = original["GAMMA_P"], original["GAMMA_F"]
    eta_p, eta_f = original["ETA_P"], original["ETA_F"]
    w, wf, delta = state["w"], state["wf"], state["Delta"]
    rp, rf, T = state["rp"], state["rf"], state["T"]

    direct_checks = []
    schedules = [(rp * q, rf * (12 - q)) for q in (0, 3, 6, 12)]
    schedules += [(137, 211), (rp * 64, rf * 96)]
    for k, ell in schedules:
        wp = np.zeros(original["D"])
        for _ in range(k):
            wp -= eta_p * (X.T @ (X @ wp - y) / n + gamma_p * wp)
        wend = wp.copy()
        for _ in range(ell):
            wend -= eta_f * (Xf.T @ (Xf @ wend - yf) / nf + gamma_f * wend)
        direct = float(hprime_norm_squared(wend - wp - delta, Xf, gamma_f))
        spectral = float(np.exp(stable(k, ell)[0]))
        relative_error = abs(spectral - direct) / direct
        np.testing.assert_allclose(spectral, direct, rtol=1e-9, atol=1e-12)
        direct_checks.append({"k": k, "ell": ell, "direct_error": direct,
                              "stable_error": spectral, "relative_difference": relative_error})

    # Full-budget 100% endpoint: the achieved update is exactly zero.
    endpoint_100_expected = float(hprime_norm_squared(delta, Xf, gamma_f))
    endpoint_100_stable = float(np.exp(stable(rp * T, 0)[0]))
    endpoint_100_saved = float(10.0 ** saved[-1])
    np.testing.assert_allclose([endpoint_100_stable, endpoint_100_saved],
                               endpoint_100_expected, rtol=1e-11, atol=0)
    np.testing.assert_allclose(stable([0, 137, rp * T], 0),
                               np.log(endpoint_100_expected), rtol=0, atol=1e-11)

    # Full-budget 0% endpoint: w_hat - C'^ell w_hat_prime. The latter
    # vector is evaluated only in the downstream ridge solution's row span.
    full_ell = rf * T
    k0_residual = w - state["Vf"] @ (state["af"] * np.exp(-full_ell * state["ratef"]))
    endpoint_0_expected = float(hprime_norm_squared(k0_residual, Xf, gamma_f))
    endpoint_0_stable = float(np.exp(stable(0, full_ell)[0]))
    endpoint_0_saved = float(10.0 ** saved[0])
    np.testing.assert_allclose([endpoint_0_stable, endpoint_0_saved],
                               endpoint_0_expected, rtol=1e-11, atol=0)

    # Verify representative saved values, both boundaries and the minimum
    # neighbourhood, independently of the original full-array sweep.
    tstar = int(np.argmin(saved))
    samples = sorted({0, T, T // 2, meta["t_lanczos_discrete"],
                      *(max(0, min(T, tstar + step)) for step in (-100, -2, -1, 0, 1, 2, 100))})
    sample_checks = []
    for t in samples:
        k, ell = rp * t, rf * (T - t)
        cached = float(saved[t])
        refreshed = float(stable(k, ell)[0] / np.log(10.0))
        independent = independent_longdouble_log10(k, ell, state, Xf, gamma_f)
        np.testing.assert_allclose(refreshed, cached, rtol=0, atol=2e-11)
        np.testing.assert_allclose(independent, cached, rtol=0, atol=2e-10)
        sample_checks.append({"t": t, "k": k, "ell": ell, "cached_log10_error": cached,
                              "same_source_log10_error": refreshed,
                              "independent_longdouble_log10_error": independent,
                              "absolute_log10_difference": abs(independent - cached)})

    pre_gradient = X.T @ (X @ w - y) / n + gamma_p * w
    fine_gradient = Xf.T @ (Xf @ wf - yf) / nf + gamma_f * wf
    assert np.linalg.norm(pre_gradient) < 1e-11
    assert np.linalg.norm(fine_gradient) < 1e-11
    return {
        "panel_index": index,
        "sample_sizes": [n, nf],
        "step_size_bounds": {
            "eta_p_lambda_max_H": float(eta_p * state["hp"][-1]),
            "eta_f_lambda_max_Hprime": float(eta_f * state["hf"][-1]),
            "both_strictly_between_zero_and_one": bool(
                0 < eta_p * state["hp"][-1] < 1 and 0 < eta_f * state["hf"][-1] < 1),
        },
        "ridge_stationarity_norms": {
            "pretraining": float(np.linalg.norm(pre_gradient)),
            "fine_tuning": float(np.linalg.norm(fine_gradient)),
        },
        "thin_svd": {
            "upstream_basis_shape": list(state["Vp"].shape),
            "downstream_basis_shape": list(state["Vf"].shape),
            "smallest_upstream_solution_coefficient": float(state["ap"][0]),
            "downstream_null_component_norm": float(np.linalg.norm(state["d0"])),
            "upstream_kernel_components": "Excluded by construction: Vp @ (ap * exp(-k * ratep)).",
        },
        "moderate_direct_gd_checks": direct_checks,
        "full_budget_endpoints": {
            "100_percent": {"k": rp * T, "ell": 0, "identity": "||Delta||^2_Hprime",
                            "independent_error": endpoint_100_expected, "stable_error": endpoint_100_stable,
                            "cached_error": endpoint_100_saved},
            "0_percent": {"k": 0, "ell": full_ell, "residual_identity": "w_hat - Cprime^ell w_hat_prime",
                          "independent_error": endpoint_0_expected, "stable_error": endpoint_0_stable,
                          "cached_error": endpoint_0_saved},
        },
        "sampled_cached_array_checks": sample_checks,
        "max_longdouble_absolute_log10_difference": max(row["absolute_log10_difference"] for row in sample_checks),
        "longdouble_check_scope": "Extended precision arithmetic on the original double-precision thin-SVD factors; full residual vector and direct design-based Hprime norm.",
    }


def main():
    original = load_original_utilities()
    metadata_path, curves_path = CACHE / "metadata.json", CACHE / "curves.npz"
    config = json.loads(metadata_path.read_text())
    expected = {
        "seed": 2026, "d": 400, "pairs": [[80, 320], [80, 80], [320, 80]],
        "base_n": 80, "budget_units": 65536, "B": 65536 * 80 * 400,
        "gamma_p": 0.05, "gamma_f": 0.05, "eta_p": 0.05, "eta_f": 0.05,
        "noise_sd": 0.10, "rotated_downstream": True, "lanczos_steps": 40,
        "objective": "squared Hprime norm",
    }
    for key, value in expected.items():
        assert config[key] == value, (key, config[key], value)
    for original_key, config_key in (
        ("SEED", "seed"), ("D", "d"), ("BASE_N", "base_n"),
        ("BUDGET_UNITS", "budget_units"), ("B", "B"), ("GAMMA_P", "gamma_p"),
        ("GAMMA_F", "gamma_f"), ("ETA_P", "eta_p"), ("ETA_F", "eta_f"),
        ("NOISE_SD", "noise_sd"), ("ROTATE_DOWNSTREAM", "rotated_downstream"),
        ("LANCZOS_STEPS", "lanczos_steps"),
    ):
        assert original[original_key] == config[config_key]
    assert [list(pair) for pair in original["PAIRS"]] == config["pairs"]

    with threadpool_limits(limits=2):
        data_dir = ROOT / "data"
        data_dir.mkdir(exist_ok=True)
        dataset_path = data_dir / "original_datasets.npz"
        dataset_archive_reused = dataset_path.exists()
        if not dataset_archive_reused:
            X, y, Xf, yf = original["make_data"]()
            generated = dict(X=X, y=y, X_prime=Xf, y_prime=yf,
                             X_80=X[:80], y_80=y[:80], X_prime_80=Xf[:80], y_prime_80=yf[:80])
            # Exclusive creation prevents an existing archive from being
            # replaced, even if another process creates it concurrently.
            with dataset_path.open("xb") as archive:
                np.savez_compressed(archive, **generated)
        archive_sha256_before_checks = sha256_file(dataset_path)
        with np.load(dataset_path, allow_pickle=False) as stored:
            datasets = {key: stored[key] for key in
                        ("X", "y", "X_prime", "y_prime", "X_80", "y_80", "X_prime_80", "y_prime_80")}
            for key, full_key in (("X_80", "X"), ("y_80", "y"),
                                  ("X_prime_80", "X_prime"), ("y_prime_80", "y_prime")):
                np.testing.assert_array_equal(datasets[key], datasets[full_key][:80])
        data = tuple(datasets[key] for key in ("X", "y", "X_prime", "y_prime"))
        X, y, Xf, yf = data
        assert X.shape == Xf.shape == (320, 400)
        assert y.shape == yf.shape == (320,)
        assert all(np.all(np.isfinite(value)) for value in data)

        array_checks, selected_checks = [], []
        with np.load(curves_path, allow_pickle=False) as cache:
            for index, meta in enumerate(config["panels"]):
                saved = cache[f"log10f_{index}"]
                n, nf = meta["n"], meta["n_prime"]
                cp, cf, budget = n * config["d"], nf * config["d"], config["B"]
                L = math.lcm(cp, cf)
                assert budget % L == 0
                T = budget // L
                t = np.arange(T + 1, dtype=np.int64)
                k, ell = (L // cp) * t, (L // cf) * (T - t)
                assert np.issubdtype(k.dtype, np.integer) and np.issubdtype(ell.dtype, np.integer)
                assert np.all(cp * k + cf * ell == budget)
                np.testing.assert_allclose(100.0 * cp * k / budget, 100.0 * t / T,
                                           rtol=0, atol=1e-12)
                assert saved.shape == (T + 1,) and np.all(np.isfinite(saved))
                assert all(meta[key] == value for key, value in
                           (("B", budget), ("T", T), ("c_p", cp), ("c_f", cf), ("L", L)))
                minimum = int(np.argmin(saved))
                assert minimum == meta["t_opt"]
                assert int(k[minimum]) == meta["k_star"] and int(ell[minimum]) == meta["ell_star"]
                percent = 100.0 * cp * int(k[minimum]) / budget
                assert percent == meta["optimum_percent"]
                values = 10.0 ** saved
                assert np.all(np.isfinite(values)) and np.all(values > 0)
                assert np.all(values >= np.finfo(float).tiny)
                assert np.count_nonzero(saved == saved[minimum]) == 1
                assert 0 < minimum < T
                assert saved[minimum] < saved[minimum - 1] and saved[minimum] < saved[minimum + 1]
                array_checks.append({
                    "panel_index": index, "plotted": index in SELECTED,
                    "n": n, "n_prime": nf, "B": budget, "T": T,
                    "c_p": cp, "c_f": cf, "L": L, "allocation_count": len(t),
                    "all_iteration_counts_integer": True, "all_budgets_equal_B": True,
                    "compute_coordinate_matches_100_t_over_T": True,
                    "argmin_t": minimum, "k_star": int(k[minimum]), "ell_star": int(ell[minimum]),
                    "optimum_percent": percent, "minimum_log10_error": float(saved[minimum]),
                    "minimum_positive_error": float(values[minimum]),
                    "maximum_log10_error": float(np.max(saved)),
                    "minimum_is_unique": True, "all_errors_finite_and_positive": True,
                    "adjacent_log10_differences": [float(saved[minimum - 1] - saved[minimum]),
                                                   float(saved[minimum + 1] - saved[minimum])],
                    "log10_distance_above_smallest_normal_float64": float(saved[minimum] - np.log10(np.finfo(float).tiny)),
                    "array": array_record(saved),
                })
                if index in SELECTED:
                    selected_checks.append(validate_selected(index, meta, saved, original, data))
        assert sha256_file(dataset_path) == archive_sha256_before_checks

    result = {
        "status": "passed", "schema": "compute_allocation_validation_v1",
        "scope": "All cached full-budget schedules and minima; selected panels receive independent residual, endpoint, stationarity, and extended-precision checks.",
        "provenance": {
            "source": str(SOURCE.relative_to(ROOT)), "source_sha256": sha256_file(SOURCE),
            "cache_metadata": str(metadata_path.relative_to(ROOT)), "cache_metadata_sha256": sha256_file(metadata_path),
            "cache_curves": str(curves_path.relative_to(ROOT)), "cache_curves_sha256": sha256_file(curves_path),
            "dataset_archive": str(dataset_path.relative_to(ROOT)), "dataset_archive_sha256": sha256_file(dataset_path),
            "dataset_archive_reused": dataset_archive_reused,
            "dataset_archive_action": "Reused existing regenerated archive without modification." if dataset_archive_reused else "Created missing archive using the supplied original make_data function and original RNG order.",
            "data_status": "The local archive was regenerated using the supplied original make_data function, original RNG call order and unchanged parameters. Subsequent validations reuse it without regeneration or replacement. No archived historical datasets or curves were supplied; bit-identical agreement with prior chapter-run arrays is not asserted.",
            "cache_status": "Generated once in this environment by the unchanged supplied original Cell 1; validation does not repeat its allocation sweep.",
            "chapter_comparison": "Section 4.5.4 agrees: independent Gaussian QR eigenbases, covariance eigenvalues geometrically spaced from 1 to 1e-3, 320 observations per phase, teacher and shift N(0,I/d), independent noise SD 0.10, first-80 nesting, stated penalties, step sizes, dimension, budget and costs.",
            "archived_historical_arrays_available": False,
        },
        "parameters": expected,
        "data_arrays": {key: array_record(value) for key, value in datasets.items()},
        "nested_first_80_checks": "Exact array equality checked for both design matrices and both label vectors.",
        "full_array_checks": array_checks,
        "selected_panel_checks": selected_checks,
        "numerical_integrity": {
            "objective_evaluator": "Exact original stable log_objective extracted by AST; exponential rescaling and upstream thin SVD preserved.",
            "objective_floor_or_clipping": "None in the source objective evaluator or saved arrays. The source clips only its unused rounded Lanczos allocation, never objective values.",
            "underflow": "Every cached log10 objective exponentiates to a finite, strictly positive normal float64 value; minima are unique and strictly lower than both adjacent allocations.",
            "false_error_floor_check": "Independent rescaled long-double residual and direct design-based norm agree at each selected minimum and neighbouring allocations, including values far below the cancellation floor of converged-iterate subtraction.",
        },
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                    "blas_thread_limit": 2, "longdouble_mantissa_bits": int(np.finfo(np.longdouble).nmant)},
    }
    output = ROOT / "outputs" / "numerical_checks.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    for panel in array_checks:
        if panel["plotted"]:
            print(f"n={panel['n']}, n'={panel['n_prime']}: {panel['allocation_count']:,} feasible schedules; "
                  f"optimum {panel['optimum_percent']:.10f}%; "
                  f"min error {panel['minimum_positive_error']:.8g}")
    print(f"All numerical checks passed. Saved {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
