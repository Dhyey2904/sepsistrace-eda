"""Shared matplotlib style and small plotting helpers."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SEPTIC = "#C4705E"
NONSEPTIC = "#8FA7C4"
BACKGROUND = "#FAF8F4"
INK = "#2B2B2B"
MUTED = "#6B6B6B"
HOSP_A = "#3D5A80"
HOSP_B = "#D9A441"
SEQ_CMAP = LinearSegmentedColormap.from_list("seq", [BACKGROUND, "#3D4F66"])
# Model / feature-set colours used by the results figures (fixed order).
SET_COLORS = {"V": "#3D5A80", "M": SEPTIC, "VM": "#6A994E"}
MODEL_COLORS = {"lr": "#3D5A80", "rf": "#6A994E", "xgb": SEPTIC, "mlp": "#8E6C8A"}


def set_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": BACKGROUND,
        "axes.facecolor": BACKGROUND,
        "savefig.facecolor": BACKGROUND,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 9,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "lines.linewidth": 2,
        "font.size": 9,
    })


def caption(fig: plt.Figure, text: str) -> None:
    """A one-line (or two-line) caption under the figure."""
    fig.text(0.01, 0.005, text, ha="left", va="bottom", fontsize=7.5, color=MUTED, style="italic", wrap=True)


def save(fig: plt.Figure, out_dir: Path, name: str, dpi: int = 300) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def first_grid(res, out_dir: Path, horizon: int, seed: int, sets=("V", "M", "VM")) -> None:
    """fig7: AUPRC-lift heatmaps (model x feature set) for internal test and hospital B."""
    import numpy as np

    SETS = list(sets)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, split in zip(axes, ("test", "external")):
        piv = res[res.split == split].pivot(index="model", columns="feature_set", values="auprc_lift")[SETS]
        auc = res[res.split == split].pivot(index="model", columns="feature_set", values="auroc")[SETS]
        im = ax.imshow(piv.values, cmap=SEQ_CMAP, vmin=0, vmax=np.nanmax(res.auprc_lift) * 1.05)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v = piv.values[i, j]
                ax.text(j, i, f"{v:.1f}×\nAUROC {auc.values[i, j]:.3f}", ha="center", va="center", fontsize=8,
                        color="white" if v > np.nanmax(res.auprc_lift) * 0.55 else INK)
        ax.set_xticks(range(len(SETS)), SETS)
        ax.set_yticks(range(len(piv.index)), piv.index)
        ax.grid(False)
        ax.set_title(f"{'Internal test (A)' if split == 'test' else 'External (hospital B)'}")
    fig.colorbar(im, ax=axes, fraction=0.025, label="AUPRC lift over chance")
    fig.suptitle(f"First results: AUPRC lift, {horizon}h horizon, seed {seed}", fontweight="bold")
    caption(fig, "Lift = AUPRC / positive rate; 1.0 is chance. V = values, M = measurement pattern only, "
                   "VM = both; all three include the six statics.")
    save(fig, out_dir, "fig7_first_grid.png")


set_style()
