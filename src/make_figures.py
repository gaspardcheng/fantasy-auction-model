"""Figures + bootstrap confidence intervals for the deck and README (results/figures/*.png)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import fantasy as F
from backtest import frame

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = RES / "figures"
OURS, BASE, LIGHT, INK, MUTED, GRID = "#2a78d6", "#8a8984", "#c9c8c2", "#0b0b0b", "#52514e", "#e6e5e0"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": GRID,
                     "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
                     "figure.dpi": 200, "savefig.bbox": "tight", "figure.facecolor": "white"})
LABEL = {"Naive (last season)": "Last season\n(rule)", "Marcel (weighted avg)": "Weighted avg\n(public rule)",
         "Ensemble Ridge+RF": "Ridge+RF\nensemble", "Our model": "Our model\n(selected)",
         "Perfect foresight": "Perfect\nforesight"}


def bars(ax, names, vals, colors, fmt, title):
    x = np.arange(len(names))
    ax.bar(x, vals, width=0.6, color=colors, edgecolor="white", linewidth=2)
    for i, v in enumerate(vals):
        ax.text(i, v, fmt(v), ha="center", va="bottom", fontsize=11, color=INK, fontweight="bold")
    ax.set_xticks(x, names)
    ax.set_title(title, loc="left", fontsize=12, color=INK, fontweight="bold")
    ax.yaxis.grid(True, color=GRID, linewidth=0.8); ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    selection = json.load(open(RES / "selection.json"))
    sel, folds = selection["selected_model"], selection["validation_folds"]
    # ---------------- business backtest
    b = pd.read_csv(RES / "backtest_summary.csv")
    b = b[b["room"] == "Marcel"].set_index("method")          # headline room: every rival on Marcel
    names = list(b.index)
    col = [OURS if n == "Our model" else (LIGHT if n == "Perfect foresight" else BASE) for n in names]
    fig, ax = plt.subplots(1, 2, figsize=(15, 4.4))
    bars(ax[0], [LABEL[n] for n in names], b["points"], col, lambda v: f"{v:.1f}", "Average roto points (max 126)")
    ax[0].axhline(67.5, color=MUTED, linestyle="--", linewidth=1)
    ax[0].text(len(names) - 0.55, 68.5, "league avg 67.5", ha="right", fontsize=9, color=MUTED)
    ax[0].set_ylim(0, 105)
    bars(ax[1], [LABEL[n] for n in names], b["win"] * 100, col, lambda v: f"{v:.0f}%", "Chance of winning the league")
    ax[1].axhline(100 / 14, color=MUTED, linestyle="--", linewidth=1)
    ax[1].text(len(names) - 0.55, 100 / 14 + 2, "1 in 14 = 7%", ha="right", fontsize=9, color=MUTED)
    ax[1].set_ylim(0, 90)
    for a in ax:
        a.tick_params(axis="x", labelsize=9)
    fig.savefig(FIG / "backtest.png"); plt.close(fig)

    # ---------------- accuracy on hold-out season
    t = pd.read_csv(RES / "test_metrics.csv").set_index("model")
    order = ["Naive (last season)", "Marcel (weighted avg)", sel]
    lab = [LABEL["Naive (last season)"], LABEL["Marcel (weighted avg)"], LABEL["Our model"]]
    col = [BASE, BASE, OURS]
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4))
    bars(ax[0], lab, t.loc[order, "Z_MAE"], col, lambda v: f"{v:.2f}", "Value per game error\n(MAE, lower is better)")
    bars(ax[1], lab, t.loc[order, "GP_MAE"], col, lambda v: f"{v:.1f}", "Games-played error\n(MAE, lower is better)")
    bars(ax[2], lab, t.loc[order, "Value_Spearman"], col, lambda v: f"{v:.2f}", "Season-value ranking\n(Spearman, higher is better)")
    for a in ax:
        a.tick_params(axis="x", labelsize=9)
    fig.savefig(FIG / "accuracy.png"); plt.close(fig)

    # ---------------- predicted vs actual (our model), with bootstrap CIs
    df = pd.read_csv(RES / "test_predictions.csv")
    act = frame(df, "actual"); played = act["GP"] >= 15
    ref = F.fit_reference(act[played])
    za = F.z_scores(act, ref)["Z"]
    preds = {m: F.z_scores(frame(df, m), ref)["Z"] for m in order}
    err = {m: (preds[m] - za)[played].abs().to_numpy() for m in order}
    gp_err = {m: (frame(df, m)["GP"] - act["GP"]).abs().to_numpy() for m in order}
    rng = np.random.default_rng(0)
    ci = {}
    for m in order[:2]:
        for k, e in (("Z", err), ("GP", gp_err)):
            d = e[sel] - e[m]
            boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
            ci[f"{k}_MAE diff vs {m}"] = {"mean": float(d.mean()), "lo": float(np.percentile(boots, 2.5)),
                                          "hi": float(np.percentile(boots, 97.5)),
                                          "pct": float(d.mean() / e[m].mean() * 100)}
    json.dump(ci, open(RES / "bootstrap_ci.json", "w"), indent=1)
    print(json.dumps(ci, indent=1))

    fig, ax = plt.subplots(figsize=(5.6, 5))
    zp = preds[sel][played]
    ax.scatter(zp, za[played], s=18, color=OURS, alpha=0.55, edgecolor="white", linewidth=0.5)
    lim = [min(zp.min(), za[played].min()) - 1, max(zp.max(), za[played].max()) + 1]
    ax.plot(lim, lim, color=MUTED, linewidth=1, linestyle="--")
    r = np.corrcoef(zp, za[played])[0, 1]
    ax.set_xlabel("Predicted fantasy value per game (z)"); ax.set_ylabel("Actual 2025-26 value per game (z)")
    ax.set_title(f"Hold-out 2025-26: r = {r:.2f}, n = {played.sum()}", loc="left", fontsize=12, color=INK, fontweight="bold")
    ax.grid(True, color=GRID, linewidth=0.8); ax.set_axisbelow(True)
    ax.spines["left"].set_visible(True); ax.spines["bottom"].set_visible(True)
    fig.savefig(FIG / "pred_vs_actual.png"); plt.close(fig)

    # ---------------- validation (model selection)
    v = pd.read_csv(RES / "validation_summary.csv").set_index("model").sort_values("Z_MAE", ascending=False)
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    colors = [OURS if m == sel else (BASE if m in order[:2] else LIGHT) for m in v.index]
    ax.scatter(v["Z_MAE"], v.index, s=90, color=colors, edgecolor="white", linewidth=1.5, zorder=3)
    for i, val in enumerate(v["Z_MAE"]):
        ax.text(val + 0.006, i, f"{val:.3f}", va="center", fontsize=9, color=INK)
    ax.set_xlim(v["Z_MAE"].min() - 0.02, v["Z_MAE"].max() + 0.04)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_title(f"Model selection: value per game MAE\n(avg of {len(folds)} folds, {folds[0]} to {folds[-1]}, lower is better)", loc="left",
                 fontsize=12, color=INK, fontweight="bold")
    ax.xaxis.grid(True, color=GRID); ax.set_axisbelow(True); ax.tick_params(axis="y", length=0, labelsize=9)
    fig.savefig(FIG / "validation.png"); plt.close(fig)
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
