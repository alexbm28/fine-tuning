from pathlib import Path
from io import BytesIO
import json, sys, shutil, subprocess
import numpy as np
import matplotlib as mpl
from matplotlib import font_manager as fm
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, FuncFormatter

# ---------------------- GRAPHICS SETTINGS ----------------------
DATA_DIR = Path("neurips_figure1_lanczos_data")
OUT_DIR = Path("neurips_figure1_lanczos")

FIG_WIDTH = 6.5
FIG_HEIGHT = 2.55
FONT_SIZE = 8
TITLE_SIZE = 9
LABEL_SIZE = 9
LEGEND_SIZE = 8

SHARE_Y = False
POWER_TICKS = True
SHOW_OPTIMUM_IN_TITLE = False
SHOW_GRID = True

X_TICKS = [0, 20, 40, 60, 80, 100]
Y_TICKS = None                         # Or 3 lists of tick exponents.
PNG_DPI = 300
DISPLAY_FIGURE = True

CURVE_STYLE = dict(
    color="#1f77b4", linewidth=1.1, linestyle="-"
)
LEADING_STYLE = dict(
    color="0.45", linewidth=0.9, linestyle="--"
)
TWO_TERM_STYLE = dict(
    color="#d62728", linewidth=1.15, linestyle=":"
)

OPTIMUM_COLOR = "black"
MARKER_SIZE = 2.2                       # Small dot; no optimum line.

# Layout dimensions in inches.
LEFT_LABEL_SPACE = 0.24
RIGHT_SPACE = 0.03
PANEL_GAP = 0.055
TOP_SPACE = 0.36
BOTTOM_SPACE = 0.34
INNER_LEFT = 0.37
INNER_RIGHT = 0.08
INNER_BOTTOM = 0.21
INNER_TOP = 0.37 if SHOW_OPTIMUM_IN_TITLE else 0.24
# ----------------------------------------------------------------

# Assemble separately drawn panels into a single vector PDF.
try:
    import pymupdf as fitz
except ImportError:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "pymupdf"],
        check=True
    )
    import pymupdf as fitz

# Match the text and mathematical fonts used in the previous figure.
if not any(f.name == "Liberation Serif" for f in fm.fontManager.ttflist):
    if shutil.which("apt-get") is None:
        raise RuntimeError("Install Liberation Serif before running this cell.")

    subprocess.run(
        ["apt-get", "update", "-qq"],
        check=True, stdout=subprocess.DEVNULL
    )
    subprocess.run(
        ["apt-get", "install", "-y", "-qq", "fonts-liberation"],
        check=True, stdout=subprocess.DEVNULL
    )
    for fontfile in fm.findSystemFonts():
        fm.fontManager.addfont(fontfile)

if not any(f.name == "Liberation Serif" for f in fm.fontManager.ttflist):
    raise RuntimeError(
        "Liberation Serif is still missing; restart after installation."
    )

if not (DATA_DIR / "metadata.json").exists():
    raise FileNotFoundError("Run the new Cell 1 first.")

config = json.loads((DATA_DIR / "metadata.json").read_text())
if config.get("schema") != "lanczos_figure1_v1":
    raise ValueError("Run the NEW Cell 1: these are not Lanczos results.")

meta = config["panels"]
with np.load(DATA_DIR / "curves.npz", allow_pickle=False) as data:
    curves = [data[f"log10f_{i}"].copy() for i in range(len(meta))]

if len(meta) != 3:
    raise ValueError("This layout expects exactly three panels.")
if (
    SHARE_Y and Y_TICKS is not None
    and any(v != Y_TICKS[0] for v in Y_TICKS)
):
    raise ValueError("Use identical Y_TICKS when SHARE_Y=True.")

OUT_DIR.mkdir(parents=True, exist_ok=True)

style = {
    "font.family": "serif",
    "font.serif": ["Liberation Serif"],
    "font.size": FONT_SIZE,
    "mathtext.fontset": "cm",
    "text.usetex": False,
    "axes.titlesize": TITLE_SIZE,
    "axes.titleweight": "normal",
    "axes.linewidth": 0.65,
    "xtick.labelsize": FONT_SIZE,
    "ytick.labelsize": FONT_SIZE,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "pdf.use14corefonts": False,
    "savefig.bbox": None,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
}


def pdf_bytes(fig):
    buffer = BytesIO()
    fig.savefig(buffer, format="pdf", bbox_inches=None)
    return buffer.getvalue()


panel_w = (
    FIG_WIDTH - LEFT_LABEL_SPACE - RIGHT_SPACE - 2*PANEL_GAP
) / 3
panel_h = FIG_HEIGHT - TOP_SPACE - BOTTOM_SPACE

if (
    panel_w <= INNER_LEFT + INNER_RIGHT
    or panel_h <= INNER_BOTTOM + INNER_TOP
):
    raise ValueError(
        "Increase figure dimensions or reduce the layout margins."
    )

with mpl.rc_context(style):
    # Shared legend and axis labels.
    frame = Figure(figsize=(FIG_WIDTH, FIG_HEIGHT))

    handles = [
        Line2D(
            [], [], **CURVE_STYLE,
            label=r"$f_{H'}(t)$"
        ),
        Line2D(
            [], [], **LEADING_STYLE,
            label=r"Lanczos: $\widehat{\theta}T$"
        ),
        Line2D(
            [], [], **TWO_TERM_STYLE,
            label=r"Two-term: $\theta^* T+s^*$"
        ),
        Line2D(
            [], [],
            color=OPTIMUM_COLOR,
            linestyle="none",
            marker="o",
            markersize=MARKER_SIZE,
            label=r"Numerical optimum $t^*$"
        ),
    ]

    frame.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.53, 0.995),
        ncol=4,
        frameon=False,
        fontsize=LEGEND_SIZE,
        columnspacing=1.4,
        handlelength=2.0,
        handletextpad=0.5
    )

    frame.text(
        0.54, 0.035,
        "Compute spent on pretraining (%)",
        ha="center", va="center", fontsize=LABEL_SIZE
    )

    ylabel = (
        r"$f_{H'}(t)=\|\Delta_t-\Delta\|_{H'}^2$"
        if POWER_TICKS
        else r"$\log_{10} f_{H'}(t)$"
    )
    frame.text(
        0.017, 0.50, ylabel,
        rotation=90, ha="center", va="center",
        fontsize=LABEL_SIZE
    )

    assembled = fitz.open(stream=pdf_bytes(frame), filetype="pdf")

    for i, (row, log10f) in enumerate(zip(meta, curves)):
        T = row["T"]
        if (
            len(log10f) != T + 1
            or not np.all(np.isfinite(log10f))
        ):
            raise ValueError(
                "The saved objective curve is incomplete or non-finite."
            )

        low = (
            min(v.min() for v in curves)
            if SHARE_Y else log10f.min()
        )
        high = (
            max(v.max() for v in curves)
            if SHARE_Y else log10f.max()
        )
        pad = 0.055 * max(high-low, 1.0)
        limits = (low-pad, high+pad)

        if Y_TICKS is None:
            ticks = MaxNLocator(
                nbins=4,
                steps=[1, 2, 2.5, 5, 10],
                integer=True
            ).tick_values(*limits)
        else:
            ticks = np.array(Y_TICKS[i])

        ticks = ticks[
            (ticks >= limits[0]) & (ticks <= limits[1])
        ]

        panel = Figure(figsize=(panel_w, panel_h))
        ax = panel.add_axes([
            INNER_LEFT/panel_w,
            INNER_BOTTOM/panel_h,
            (panel_w-INNER_LEFT-INNER_RIGHT)/panel_w,
            (panel_h-INNER_BOTTOM-INNER_TOP)/panel_h
        ])

        # Plot stored logarithms directly to avoid underflow.
        ax.plot(
            100*np.arange(T+1)/T, log10f,
            **CURVE_STYLE, zorder=2
        )

        # This field is the independent Lanczos estimate, NOT exact theta.
        ax.axvline(
            row["lanczos_percent"],
            **LEADING_STYLE, zorder=3
        )

        if 0 <= row["two_term_percent"] <= 100:
            ax.axvline(
                row["two_term_percent"],
                **TWO_TERM_STYLE, zorder=4
            )
        else:
            print(
                f"Panel {i+1}: two-term prediction is outside "
                "[0,100]%; not clipped."
            )

        # Numerical optimum: small black dot only.
        ax.plot(
            row["optimum_percent"],
            log10f[row["t_opt"]],
            marker="o",
            color=OPTIMUM_COLOR,
            markersize=MARKER_SIZE,
            linestyle="none",
            zorder=5
        )

        ax.set_xlim(0, 100)
        ax.set_ylim(*limits)
        ax.set_xticks(X_TICKS)
        ax.set_yticks(ticks)

        if POWER_TICKS:
            ax.yaxis.set_major_formatter(
                FuncFormatter(lambda x, _: rf"$10^{{{x:g}}}$")
            )

        ax.tick_params(axis="both", pad=2)
        if SHARE_Y and i > 0:
            ax.tick_params(axis="y", labelleft=False)

        ax.set_axisbelow(True)
        if SHOW_GRID:
            ax.grid(True, linewidth=0.4, alpha=0.25)

        title = rf"$n={row['n']},\quad n'={row['n_prime']}$"
        if SHOW_OPTIMUM_IN_TITLE:
            title += (
                "\n"
                + rf"$t^*/T\approx {row['optimum_percent']/100:.3f}$"
            )
        ax.set_title(title, pad=5)

        # Assemble into one row, keeping vector text and curves.
        left = LEFT_LABEL_SPACE + i*(panel_w + PANEL_GAP)
        box = fitz.Rect(
            72*left,
            72*TOP_SPACE,
            72*(left+panel_w),
            72*(TOP_SPACE+panel_h)
        )
        with fitz.open(
            stream=pdf_bytes(panel), filetype="pdf"
        ) as source:
            assembled[0].show_pdf_page(box, source, 0)

    assembled.save(
        OUT_DIR / "figure1_lanczos.pdf",
        garbage=4, deflate=True
    )
    assembled[0].get_pixmap(dpi=PNG_DPI).save(
        OUT_DIR / "figure1_lanczos.png"
    )
    (OUT_DIR / "figure1_lanczos.svg").write_text(
        assembled[0].get_svg_image(text_as_path=True)
    )
    assembled.close()

# Save the numerical results with the figure.
shutil.copytree(DATA_DIR, OUT_DIR / "data", dirs_exist_ok=True)
archive = shutil.make_archive(
    str(OUT_DIR), "zip", root_dir=OUT_DIR
)

if DISPLAY_FIGURE:
    from IPython.display import Image, display
    display(
        Image(
            filename=str(OUT_DIR / "figure1_lanczos.png"),
            width=1000
        )
    )

print(
    "Grey dashed: independent Lanczos estimate; "
    "red dotted: exact-spectrum two-term prediction."
)
print("Numerical optimum: small black dot only, including in the legend.")
print("Saved PDF, SVG, PNG and numerical data to", archive)
print(
    "Y-axis:",
    "shared range"
    if SHARE_Y
    else "separate ranges; same objective in every panel"
)