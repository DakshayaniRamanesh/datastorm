"""
EDA Notebook (Script Form) - Exploratory Data Analysis
=======================================================
Produces visualisations and summary statistics for the PDF report.
Run after Silver cleaning.

Usage:
    python pipeline/05_eda.py
"""

import sys
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

warnings.filterwarnings("ignore")
plt.style.use("dark_background")

ROOT    = Path(__file__).parent.parent
SILVER  = ROOT / "pipeline" / "silver"
GOLD    = ROOT / "pipeline" / "gold"
OUTPUT  = ROOT / "output"
OUTPUT.mkdir(parents=True, exist_ok=True)


def load_data():
    tx     = pd.read_parquet(SILVER / "transactions.parquet")
    outlet = pd.read_parquet(SILVER / "outlet_master.parquet")
    coords = pd.read_parquet(SILVER / "outlet_coordinates.parquet")
    return tx, outlet, coords


def plot_volume_distribution(tx, ax):
    vols = np.log1p(tx["Volume_Liters"])
    ax.hist(vols, bins=80, color="#00d4ff", alpha=0.8, edgecolor="none")
    ax.set_title("Log Volume Distribution (Clean Transactions)", fontsize=11, color="white")
    ax.set_xlabel("log(1 + Volume_Liters)")
    ax.set_ylabel("Frequency")


def plot_monthly_trend(tx, ax):
    monthly = tx.groupby(["Year", "Month"])["Volume_Liters"].sum().reset_index()
    monthly["period"] = monthly["Year"].astype(str) + "-" + monthly["Month"].astype(str).str.zfill(2)
    ax.plot(range(len(monthly)), monthly["Volume_Liters"] / 1e6,
            color="#ff6b6b", linewidth=2, marker="o", markersize=3)
    ax.set_title("Total Network Volume by Month (M Liters)", fontsize=11, color="white")
    ax.set_xlabel("Period (2023-2025)")
    ax.set_ylabel("Volume (M L)")
    tick_positions = list(range(0, len(monthly), 6))
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([monthly["period"].iloc[i] for i in tick_positions], rotation=45)


def plot_outlet_type_dist(outlet, ax):
    counts = outlet["Outlet_Type"].value_counts()
    colors = plt.cm.Set2(np.linspace(0, 1, len(counts)))
    ax.barh(counts.index, counts.values, color=colors)
    ax.set_title("Outlet Type Distribution", fontsize=11, color="white")
    ax.set_xlabel("Count")


def plot_volume_by_type(tx, outlet, ax):
    merged = tx.merge(outlet[["Outlet_ID","Outlet_Type"]], on="Outlet_ID")
    type_vol = merged.groupby("Outlet_Type")["Volume_Liters"].median().sort_values(ascending=False)
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(type_vol)))
    ax.bar(type_vol.index, type_vol.values, color=colors)
    ax.set_title("Median Monthly Volume by Outlet Type", fontsize=11, color="white")
    ax.set_xlabel("Outlet Type")
    ax.set_ylabel("Median Vol (L)")
    ax.tick_params(axis="x", rotation=45)


def plot_geo_distribution(coords, ax):
    ax.scatter(coords["Longitude"], coords["Latitude"], s=0.5,
               alpha=0.3, color="#00ff88")
    ax.set_title("Outlet Geospatial Distribution (Sri Lanka)", fontsize=11, color="white")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")


def plot_censoring_dist(ax):
    try:
        gold = pd.read_parquet(GOLD / "gold_features.parquet")
        ax.hist(gold["censoring_score"], bins=40, color="#ffd700", alpha=0.8, edgecolor="none")
        ax.axvline(0.4, color="#ff4444", linestyle="--", label="Constrained threshold")
        ax.set_title("Censoring Score Distribution", fontsize=11, color="white")
        ax.set_xlabel("Censoring Score")
        ax.set_ylabel("Outlets")
        ax.legend(fontsize=8)
    except Exception:
        ax.text(0.5, 0.5, "Run 04_gold first", ha="center", va="center", color="white")


def main():
    print("=" * 60)
    print("EDA - Generating Analysis Plots")
    print("=" * 60)

    tx, outlet, coords = load_data()
    monthly = tx.groupby(["Outlet_ID","Year","Month"])["Volume_Liters"].sum().reset_index()
    monthly.rename(columns={"Volume_Liters":"monthly_volume"}, inplace=True)

    fig = plt.figure(figsize=(18, 12), facecolor="#0d1117")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])

    for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")

    plot_volume_distribution(tx, ax1)
    plot_monthly_trend(tx, ax2)
    plot_outlet_type_dist(outlet, ax3)
    plot_volume_by_type(tx, outlet, ax4)
    plot_geo_distribution(coords, ax5)
    plot_censoring_dist(ax6)

    fig.suptitle("DataStorm 2026 - EDA Dashboard", fontsize=16, color="white",
                 fontweight="bold", y=0.98)

    out_path = OUTPUT / "eda_dashboard.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    print(f"  Saved: {out_path}")

    # Print summary table
    print("\n[STATS] Key Statistics:")
    print(f"  Clean transaction rows:  {len(tx):,}")
    print(f"  Unique outlets:          {tx.Outlet_ID.nunique():,}")
    print(f"  Date range:              2023-01 to 2025-12")
    print(f"  Volume range:            {tx.Volume_Liters.min():.2f} - {tx.Volume_Liters.max():.2f} L")
    print(f"  Median monthly vol/outlet: {monthly.groupby('Outlet_ID')['monthly_volume'].median().median():.2f} L")

    try:
        gold = pd.read_parquet(GOLD / "gold_features.parquet")
        n_constrained = (gold["censoring_score"] > 0.4).sum()
        print(f"\n  Outlets flagged as likely constrained: {n_constrained:,} "
              f"({100*n_constrained/len(gold):.1f}%)")
        print(f"  Avg potential multiplier: {gold['potential_multiplier'].mean():.2f}×")
        print(f"  Max potential multiplier: {gold['potential_multiplier'].max():.2f}×")
    except Exception:
        pass

    print("\n[OK]  EDA complete.\n")


if __name__ == "__main__":
    main()
