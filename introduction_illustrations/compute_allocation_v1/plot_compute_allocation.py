#!/usr/bin/env python3
"""Replot the cached original Figure 4.1 experiment; never generate data.

Run from any directory: python plot_compute_allocation.py
Outputs are PDF/SVG vectors and a 600 dpi PNG at the chapter's 150 mm
text width. The source cache and supplied experiment scripts are read-only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "compute-allocation-mpl"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, LogFormatterMathtext, NullLocator
import numpy as np

ROOT = Path(__file__).resolve().parent
SELECTED = ((80, 320), (320, 80))
WIDTH_MM = 150.0
HEIGHT_MM = 98.0
BLUE = "#1f77b4"
INK = "#202020"
MATH_ERROR = r"$\|\boldsymbol{\Delta}_{k,\ell}-\boldsymbol{\Delta}\|_{H'}^2$"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_panels(data_dir):
    metadata_path = data_dir / "metadata.json"
    curves_path = data_dir / "curves.npz"
    if not metadata_path.is_file() or not curves_path.is_file():
        raise FileNotFoundError("Supply the original metadata.json and curves.npz; this script does not simulate replacement data.")
    config = json.loads(metadata_path.read_text())
    expected = dict(schema="lanczos_figure1_v1", seed=2026, d=400,
                    base_n=80, budget_units=65536, B=65536*80*400,
                    gamma_p=0.05, gamma_f=0.05, eta_p=0.05, eta_f=0.05,
                    noise_sd=0.10, rotated_downstream=True,
                    objective="squared Hprime norm")
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"Unexpected original experiment setting {key}: {config.get(key)!r}")
    panels = []
    with np.load(curves_path, allow_pickle=False) as arrays:
        for n, nf in SELECTED:
            matches = [(i, p) for i, p in enumerate(config["panels"])
                       if (p["n"], p["n_prime"]) == (n, nf)]
            if len(matches) != 1:
                raise ValueError(f"Expected one saved panel for {(n, nf)}")
            index, row = matches[0]
            log10f = arrays[f"log10f_{index}"].copy()
            B, cp, cf = config["B"], n*config["d"], nf*config["d"]
            L = math.lcm(cp, cf)
            if B % L:
                raise ValueError("The budget is not compatible with integer schedules")
            T = B // L
            for key, value in dict(B=B, c_p=cp, c_f=cf, L=L, T=T).items():
                if row[key] != value:
                    raise ValueError(f"Inconsistent panel metadata: {key}")
            if log10f.shape != (T+1,) or not np.all(np.isfinite(log10f)):
                raise ValueError("Saved curve is incomplete or nonfinite")
            t = np.arange(T+1, dtype=np.int64)
            k, ell = (L//cp)*t, (L//cf)*(T-t)
            np.testing.assert_array_equal(cp*k + cf*ell, np.full(T+1, B))
            x = 100.0*cp*k/B
            np.testing.assert_allclose(x, 100.0*t/T, rtol=0, atol=1e-12)
            tstar = int(np.argmin(log10f))
            percent = float(x[tstar])
            if tstar != row["t_opt"] or not np.isclose(percent, row["optimum_percent"], rtol=0, atol=1e-12):
                raise ValueError("Saved numerical minimiser disagrees with the complete curve")
            # Both selected curves are safely representable in float64.
            # Use an actual logarithmic Matplotlib axis, without flooring or
            # normalising errors. This conversion does not rerun GD.
            with np.errstate(under="raise", over="raise", invalid="raise"):
                values = np.power(10.0, log10f)
            if not np.all(np.isfinite(values) & (values > 0)):
                raise FloatingPointError("Original positive errors cannot be represented without loss")
            np.testing.assert_allclose(np.log10(values), log10f, rtol=0, atol=1e-13)
            panels.append(dict(index=index, meta=row, x=x, values=values,
                               log10f=log10f, tstar=tstar, percent=percent,
                               kstar=int(k[tstar]), ellstar=int(ell[tstar])))
    return config, panels


def figure_style():
    if not any(f.name == "Liberation Serif" for f in font_manager.fontManager.ttflist):
        raise RuntimeError("Install Liberation Serif (fonts-liberation) to match the supplied plot typography.")
    return {
        "font.family": "serif", "font.serif": ["Liberation Serif"],
        "font.size": 9, "mathtext.fontset": "cm", "text.usetex": False,
        "text.color": INK, "axes.labelcolor": INK,
        "axes.linewidth": 0.65, "axes.edgecolor": "0.35",
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "path",
        "pdf.use14corefonts": False, "figure.facecolor": "white",
        "axes.facecolor": "white", "savefig.bbox": None,
        # Retain all 16,385 original points in each exported curve.
        "path.simplify": False, "agg.path.chunksize": 0,
    }


def draw_schematic(fig):
    ax = fig.add_axes([0, 0, 1, 1], frameon=False)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.055, 0.958, "A", fontweight="bold", fontsize=10, va="center")
    ax.text(0.089, 0.958, "Two-stage training", fontsize=9.5, va="center")
    ax.text(0.967, 0.958, r"Shared budget: $c_p k+c_f\ell=B$",
            fontsize=9, ha="right", va="center")
    y = 0.825
    for x, label in [(0.105, r"$\mathbf{0}$"),
                     (0.425, r"$\mathbf{w}_k$"),
                     (0.890, r"$\mathbf{w}'_{\ell}$")]:
        ax.text(x, y, label, fontsize=13, ha="center", va="center")
    arrows = [(0.145, 0.379, r"Pretraining: $k$ steps"),
              (0.478, 0.836, r"Fine-tuning: $\ell$ steps")]
    for start, end, title in arrows:
        ax.annotate("", xy=(end, y), xytext=(start, y),
                    arrowprops=dict(arrowstyle="->", lw=1.0, color="0.30", mutation_scale=10))
        ax.text((start+end)/2, 0.880, title, ha="center", va="center", fontsize=9)
    # The bracket explicitly starts at w_k and ends at w'_ell. It is a
    # procedural annotation, not a coordinate-space trajectory or time scale.
    ax.plot([0.425, 0.425, 0.890, 0.890], [0.782, 0.762, 0.762, 0.782],
            color="0.48", lw=0.7, clip_on=False)
    ax.text(0.6575, 0.716,
            r"Achieved update: $\boldsymbol{\Delta}_{k,\ell}=\mathbf{w}'_\ell-\mathbf{w}_k$",
            ha="center", va="center", fontsize=9)
    ax.text(0.055, 0.716,
            r"Ideal update: $\boldsymbol{\Delta}=\widehat{\mathbf{w}}'-\widehat{\mathbf{w}}$",
            ha="left", va="center", fontsize=9)
    ax.plot([0.055, 0.978], [0.649, 0.649], color="0.87", lw=0.65)
    return ax


def draw_figure(panels, output_dir):
    low = min(float(p["log10f"].min()) for p in panels)
    high = max(float(p["log10f"].max()) for p in panels)
    padding = 0.05*(high-low)
    limits_log10 = (low-padding, high+padding)
    limits = tuple(10.0**v for v in limits_log10)
    with plt.rc_context(figure_style()):
        fig = plt.figure(figsize=(WIDTH_MM/25.4, HEIGHT_MM/25.4), dpi=150)
        draw_schematic(fig)
        axes, annotations = [], []
        bounds = [[0.133, 0.148, 0.376, 0.414], [0.601, 0.148, 0.376, 0.414]]
        for i, (panel, box) in enumerate(zip(panels, bounds)):
            ax = fig.add_axes(box, sharey=axes[0] if axes else None)
            axes.append(ax)
            row = panel["meta"]
            ax.plot(panel["x"], panel["values"], color=BLUE, lw=1.15, zorder=3,
                    solid_joinstyle="round", label="Numerical update error")
            ax.axvline(panel["percent"], color="0.66", lw=0.8, linestyle=(0, (3, 3)), zorder=1)
            ax.plot(panel["percent"], panel["values"][panel["tstar"]],
                    "o", color="black", ms=4.0, markeredgecolor="white",
                    markeredgewidth=0.4, zorder=5)
            ax.set_yscale("log", nonpositive="mask")
            ax.set_xlim(0, 100)
            ax.set_ylim(*limits)
            ax.set_xticks([0, 20, 40, 60, 80, 100])
            ticks = 10.0**np.arange(-30, 1, 10, dtype=float)
            ax.yaxis.set_major_locator(FixedLocator(ticks))
            ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
            ax.yaxis.set_minor_locator(NullLocator())
            ax.tick_params(axis="both", pad=3)
            ax.grid(axis="y", color="0.91", linewidth=0.45, zorder=0)
            ax.set_axisbelow(True)
            ax.text(0, 1.105, chr(ord("B")+i), transform=ax.transAxes,
                    fontweight="bold", fontsize=10, va="center")
            ax.text(0.10, 1.105, rf"$n={row['n']},\quad n'={row['n_prime']}$",
                    transform=ax.transAxes, fontsize=10, va="center")
            # Anchor the label in open space, to the right of the left-hand
            # optimum and left of the right-hand optimum. Its percentage is
            # calculated from argmin over every saved feasible schedule.
            label_xy = (0.32, 0.16) if i == 0 else (0.47, 0.23)
            label = ax.annotate(f"Best split: {panel['percent']:.2f}%",
                                xy=(panel["percent"], panel["values"][panel["tstar"]]),
                                xycoords="data", xytext=label_xy, textcoords="axes fraction",
                                ha="left" if i == 0 else "right", va="center", fontsize=8.5,
                                arrowprops=dict(arrowstyle="-", color="0.32", lw=0.65,
                                                shrinkA=3, shrinkB=5),
                                bbox=dict(boxstyle="square,pad=0.12", fc="white", ec="none"),
                                zorder=6)
            annotations.append(label)
        fig.text(0.035, 0.355, "Squared update error\n"+MATH_ERROR,
                 rotation=90, ha="center", va="center", fontsize=9, linespacing=1.5)
        fig.text(0.555, 0.044, "Compute spent on pretraining (%)",
                 ha="center", va="center", fontsize=9)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        label_boxes = []
        for ax, label in zip(axes, annotations):
            # get_bbox_patch excludes the annotation leader, so this check
            # tests the text box against the actual numerical curve.
            box = label.get_bbox_patch().get_window_extent(renderer)
            xy = ax.transData.transform(np.column_stack((ax.lines[0].get_xdata(), ax.lines[0].get_ydata())))
            inside = ((xy[:, 0] >= box.x0-1) & (xy[:, 0] <= box.x1+1)
                      & (xy[:, 1] >= box.y0-1) & (xy[:, 1] <= box.y1+1))
            if np.any(inside):
                raise RuntimeError("An optimum label overlaps the numerical curve")
            label_boxes.append([float(v) for v in box.bounds])
        for extension in ("pdf", "svg", "png"):
            kwargs = dict(dpi=600, bbox_inches=None)
            if extension == "pdf":
                kwargs["metadata"] = {"Title": "Compute allocation: When to switch training between objectives?",
                                      "Creator": "Matplotlib; original Figure 4.1 numerical cache",
                                      "CreationDate": None, "ModDate": None}
            elif extension == "svg":
                kwargs["metadata"] = {"Title": "Compute allocation under a shared training budget", "Date": None}
            fig.savefig(output_dir/f"compute_allocation.{extension}", **kwargs)
        # A smaller inspection copy at 150 dpi; not substituted for the 600 dpi export.
        fig.savefig(output_dir/"inspection_150dpi.png", dpi=150, bbox_inches=None)
        report = {"figure_width_mm": WIDTH_MM, "figure_height_mm": HEIGHT_MM,
                  "font": "Liberation Serif; Computer Modern mathematics", "png_dpi": 600,
                  "x_limits": [0, 100], "shared_log10_y_limits": list(limits_log10),
                  "normalisation": "None", "clipping_or_flooring": "None",
                  "curve_path_simplification": False,
                  "optimum_label_curve_intersections": 0,
                  "label_bounds_display_pixels": label_boxes,
                  "axis_scales": [a.get_yscale() for a in axes]}
        plt.close(fig)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT/"neurips_figure1_lanczos_data")
    parser.add_argument("--output-dir", type=Path, default=ROOT/"outputs")
    args = parser.parse_args()
    config, panels = load_panels(args.data_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = draw_figure(panels, args.output_dir)
    report["source_metadata_sha256"] = digest(args.data_dir/"metadata.json")
    report["source_curves_sha256"] = digest(args.data_dir/"curves.npz")
    report["provenance"] = "Replotted cached results regenerated once from the supplied unchanged original Cell 1. Historical saved arrays were not supplied."
    report["runtime"] = dict(python=platform.python_version(), numpy=np.__version__, matplotlib=matplotlib.__version__)
    report["panels"] = [dict(n=p["meta"]["n"], n_prime=p["meta"]["n_prime"],
                            cache_panel_index=p["index"], feasible_points=len(p["x"]),
                            t_opt=p["tstar"], k_star=p["kstar"], ell_star=p["ellstar"],
                            optimum_percent=p["percent"], displayed_percent=f"{p['percent']:.2f}%",
                            minimum_squared_update_error=float(p["values"][p["tstar"]]),
                            minimum_log10_squared_update_error=float(p["log10f"][p["tstar"]]))
                        for p in panels]
    (args.output_dir/"plot_record.json").write_text(json.dumps(report, indent=2)+"\n")
    for p in report["panels"]:
        print(f"({p['n']}, {p['n_prime']}): optimum {p['optimum_percent']:.10f}% across {p['feasible_points']:,} feasible allocations")
    print(f"Saved PDF, SVG and 600 dpi PNG in {args.output_dir}")


if __name__ == "__main__":
    main()
