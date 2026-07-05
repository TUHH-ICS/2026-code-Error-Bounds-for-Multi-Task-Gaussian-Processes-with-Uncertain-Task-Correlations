#!/usr/bin/env python3

import argparse
import os
from pathlib import Path

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib-cache"
    os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import seaborn as sns
import torch


EXPERIMENTS = {
    "Mean": "mean2_all.pt",
    "Stochastic": "stochastic2_all.pt",
    "Sample": "sample3_all.pt",
    "Extra": "extra_all.pt",
}

METRIC_LABELS = {
    "rate": "Failure rate",
    "T": "Mean scaled distance",
    "failures": "Failures",
    "nu": r"$\nu$",
}


def scalar(value):
    if torch.is_tensor(value):
        return float(value.detach().cpu().reshape(-1).mean())
    return float(value)


def load_experiment(summary_dir, label, filename, metric):
    path = Path(summary_dir) / filename
    if not path.exists():
        raise FileNotFoundError(path)

    data = torch.load(path, map_location="cpu", weights_only=False)
    results = data["results"] if isinstance(data, dict) else data

    rows = []
    for result in results:
        rows.append(
            {
                "experiment": label,
                "rho": scalar(result["rho"]),
                "seed": int(result["seed"]),
                "value": max(scalar(result[metric]),0.001),
            }
        )
    return rows


def load_dataframe(summary_dir, metric, excluded_rhos=()):
    rows = []
    for label, filename in EXPERIMENTS.items():
        rows.extend(load_experiment(summary_dir, label, filename, metric))

    df = pd.DataFrame(rows)
    for rho in excluded_rhos:
        df = df[(df["rho"] - rho).abs() > 1e-12]
    if df.empty:
        raise RuntimeError("No rows left after excluding rho values.")

    rho_order = sorted(df["rho"].unique())
    df["rho_label"] = pd.Categorical(
        df["rho"].map(lambda rho: f"{rho:g}"),
        categories=[f"{rho:g}" for rho in rho_order],
        ordered=True,
    )
    return df, rho_order


def configure_plotting():
    sns.set_theme(context="paper", style="whitegrid", font_scale=1.15)
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )


def add_rate_targets(ax, rho_order):
    for idx, rho in enumerate(rho_order):
        ax.scatter(idx, rho, marker="_", s=260, color="black", linewidths=1.6, zorder=5)
    return Line2D([0], [0], marker="_", color="black", linestyle="None", markersize=12)


def make_plot(df, rho_order, metric, show_targets):
    configure_plotting()
    fig, ax = plt.subplots(figsize=(9.0, 4.2), constrained_layout=True)

    sns.violinplot(
        data=df,
        x="rho_label",
        y="value",
        hue="experiment",
        order=[f"{rho:g}" for rho in rho_order],
        hue_order=list(EXPERIMENTS),
        palette="colorblind",
        bw_adjust=0.5,
        inner=None,
        cut=0,
        linewidth=0.9,
        density_norm="width",
        common_norm=True,
        ax=ax,
        log_scale=10. if metric in {"rate", "failures"} else 2,
    )

    ax.set_xlabel(r"Target $\rho$")
    ax.set_ylabel(METRIC_LABELS[metric])
    if metric in {"rate", "failures", "T"}:
        if ax.get_yscale() == "log":
            ax.set_ylim(bottom=max(df["value"].min() * 0.8, 1e-12))
        else:
            ax.set_ylim(bottom=0)

    handles, labels = ax.get_legend_handles_labels()
    if metric == "rate" and show_targets:
        plt.hlines([rho for rho in rho_order], *ax.get_xlim(), colors="black", linestyles="dotted", linewidth=0.8, zorder=4,alpha=0.5, )
        [plt.text(-.4,rho,f"{rho}", ha="center", va="bottom") for rho in rho_order]
    ax.legend(handles, labels, loc="best", ncol=min(len(labels), 4))
    sns.despine(fig=fig, ax=ax, trim=True)
    return fig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-dir", default="data/summary")
    parser.add_argument("--metric", choices=METRIC_LABELS, default="rate")
    parser.add_argument("--out-dir", default="plots")
    parser.add_argument("--out-name", default=None)
    parser.add_argument("--exclude-rho", action="append", type=float, default=[0.5])
    parser.add_argument("--no-targets", action="store_true")
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()

# import tikzplotlib
def main():
    args = parse_args()
    df, rho_order = load_dataframe(args.summary_dir, args.metric, args.exclude_rho)
    fig = make_plot(df, rho_order, args.metric, show_targets=not args.no_targets)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = args.out_name or f"conservatism_violin_{args.metric}"

    for suffix in ("pdf", "png"):
        path = out_dir / f"{out_name}.{suffix}"
        fig.savefig(path, bbox_inches="tight")
        # tikzplotlib.save(out_dir / f"{out_name}.tex")
        print(f"Saved {path}")

    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
