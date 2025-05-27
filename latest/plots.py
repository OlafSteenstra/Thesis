#!/usr/bin/env python3
# ------------------------------------------------------------
#  generate_plots.py
#
#  Produce the ten evaluation figures listed in the plan and
#  save them to disk.  Works directly from the “results.csv”
#  created earlier (columns assumed: vertices, retic_ratio,
#  algo_time, exact_time, post_ratio, …).
# ------------------------------------------------------------
import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from matplotlib.ticker import LogLocator, NullFormatter


# ── CLI & directories ─────────────────────────────────────────
csv_file = sys.argv[1] if len(sys.argv) > 1 else "results.csv"
out_dir  = sys.argv[2] if len(sys.argv) > 2 else "plots"
os.makedirs(out_dir, exist_ok=True)

# ── Load & enrich data ────────────────────────────────────────
df = pd.read_csv(csv_file)

# Derived metrics
df["speedup"]      = df["exact_time"] / df["algo_time"]
df["approx_ratio"] = df["post_ratio"]                # pick post-ratio as “approximation ratio”
df["solved"]       = ~df["exact_time"].isna()        # MPNet solved flag
df["retics"]       = df["retic_ratio"] * df["vertices"]  # retics = reticulation ratio * vertices
df["diff"]         = df["pre_result"] - df["post_result"]  # difference between exact and approx. results
df["diff_ratio"]   = df["diff"] / df["exact_result"]  # difference ratio

# Reticulation & vertex bins for grouped plots
df["retic_bin"] = pd.cut(
    df["retic_ratio"],
    bins=[-0.01, 0.05, 0.10, 0.20, 0.40, 1.01],
    labels=["0-5 %", "5-10 %", "10-20 %", "20-40 %", "40-100 %"]
)
df["vert_bin"] = pd.cut(
    df["vertices"],
    bins=[1, 1000, 2000, 3000, 4000, np.inf],
    labels=["<1k", "1k-2k", "2k-3k", "3k-4k", ">4k"]
)

# Consistent seaborn style
sns.set_style("whitegrid")
plt.rcParams["savefig.dpi"] = 300

plt.figure(figsize=(6, 4))
sns.scatterplot(data=df, x="vertices", y="diff")
plt.axhline(0, ls="--", c="grey", lw=.8)
plt.xlabel("Internal nodes")
plt.ylabel("Difference in SPS")
plt.title("Difference between pre- and post-improvement SPS")
plt.tight_layout()
plt.savefig(f"{out_dir}/diff_vs_size.png")
plt.close()

plt.figure(figsize=(6, 4))
sns.scatterplot(data=df, x="retics", y="diff")
plt.axhline(0, ls="--", c="grey", lw=.8)
plt.xlabel("Reticulation nodes")
plt.ylabel("Difference in SPS")
plt.title("Difference between pre- and post-improvement SPS")
plt.tight_layout()
plt.savefig(f"{out_dir}/diff_vs_retics.png")
plt.close()

plt.figure(figsize=(6, 4))
sns.scatterplot(data=df, x="vertices", y="diff_ratio")
plt.axhline(0, ls="--", c="grey", lw=.8)
plt.xlabel("Internal nodes")
plt.ylabel("Difference in approximation ratio")
plt.title("Difference between pre- and post-improvement approximation ratio")
plt.tight_layout()
plt.savefig(f"{out_dir}/diff_r_vs_size.png")
plt.close()

plt.figure(figsize=(6, 4))
sns.scatterplot(data=df, x="retics", y="diff_ratio")
plt.axhline(0, ls="--", c="grey", lw=.8)
plt.xlabel("Reticulation nodes")
plt.ylabel("Difference in approximation ratio")
plt.title("Difference between pre- and post-improvement approximation ratio")
plt.tight_layout()
plt.savefig(f"{out_dir}/diff_r_vs_retics.png")
plt.close()
# ───────────────── 1.  Runtime vs vertices ───────────────────
plt.figure(figsize=(6, 4))
sns.scatterplot(data=df, x="vertices", y="exact_time",  label="Exact (MPNet)", marker="o")
sns.scatterplot(data=df, x="vertices", y="algo_time",   label="Approx.",       marker="s")
plt.yscale("log")
plt.xlabel("Internal nodes")
plt.ylabel("Runtime (s, log)")
plt.title("Runtime vs graph size")
plt.tight_layout()
plt.legend()
plt.savefig(f"{out_dir}/runtime_vs_size.png")
plt.close()

# ───────────────── 2.  Runtime vs reticulation ───────────────
plt.figure(figsize=(6, 4))
sns.scatterplot(data=df, x="retic_ratio", y="exact_time", label="Exact (MPNet)", marker="o")
sns.scatterplot(data=df, x="retic_ratio", y="algo_time",  label="Approx.",       marker="s")
plt.yscale("log")
plt.xlabel("Reticulation ratio")
plt.ylabel("Runtime (s, log)")
plt.title("Runtime vs reticulation")
plt.tight_layout()
plt.legend()
plt.savefig(f"{out_dir}/runtime_vs_retic.png")
plt.close()

plt.figure(figsize=(6, 4))
sns.scatterplot(data=df, x="retics", y="exact_time", label="Exact (MPNet)", marker="o")
sns.scatterplot(data=df, x="retics", y="algo_time",  label="Approx.",       marker="s")
plt.yscale("log")
plt.xlabel("Reticulation nodes")
plt.ylabel("Runtime (s, log)")
plt.title("Runtime vs reticulation")
plt.tight_layout()
plt.legend()
plt.savefig(f"{out_dir}/runtime_vs_retics.png")
plt.close()

# ───────────────── 3.  Speed-up curves ───────────────────────
for xcol, name in [("vertices", "size"), ("retic_ratio", "retic"), ("retics", "retics")]:	
    plt.figure(figsize=(6, 4))

    # ── raw points ─────────────────────────────────────────────
    sns.scatterplot(data=df, x=xcol, y="speedup", alpha=0.7)

    # ── trend line: LOWESS (non-parametric) ───────────────────
    #    scatter=False prevents the points from being re-plotted
    sns.regplot(
        data=df,
        x=xcol,
        y="speedup",
        scatter=False,
        lowess=True,               # robust locally-weighted smoother
        color="black",
        line_kws={"lw": 1.2},
    )

    # ── axis scales & labels ──────────────────────────────────
    if xcol == "vertices":
        plt.xlabel("Internal nodes")
    elif xcol == "retic_ratio":
        plt.xlabel("Reticulation ratio")
    else:
        plt.xlabel("Reticulation nodes")

    plt.yscale("log")
    plt.ylabel("Speed-up  (exact / approx, log)")
    plt.title(f"Algorithm speed-up vs {name}")

    plt.tight_layout()
    plt.savefig(f"{out_dir}/speedup_vs_{name}.png")
    plt.close()

# ───────────────── 4.  Approx-ratio histogram & ECDF ─────────
plt.figure(figsize=(6, 4))
sns.histplot(df["approx_ratio"].dropna(), bins=30)
plt.xlabel("Approximation ratio")
plt.title("Distribution of approximation ratios")
plt.tight_layout()
plt.savefig(f"{out_dir}/ratio_histogram.png")
plt.close()

plt.figure(figsize=(6, 4))
sns.ecdfplot(df["approx_ratio"].dropna())
plt.xlabel("Approximation ratio")
plt.ylabel("Cumulative fraction")
plt.title("ECDF of approximation ratios")
plt.tight_layout()
plt.savefig(f"{out_dir}/ratio_ecdf.png")
plt.close()

# ───────────────── 5.  Ratio vs retic (scatter + LOWESS) ─────
plt.figure(figsize=(6, 4))
sns.scatterplot(data=df, x="retic_ratio", y="approx_ratio", hue="vert_bin", alpha=.7)
sns.regplot(data=df, x="retic_ratio", y="approx_ratio", scatter=False, lowess=True, color="black")
plt.axhline(1, ls="--", c="grey", lw=.8)
plt.xlabel("Reticulation ratio")
plt.ylabel("Approximation ratio")
plt.title("Approximation ratio vs reticulation")
plt.legend(title="Internal nodes", loc="upper left")
plt.tight_layout()
plt.savefig(f"{out_dir}/ratio_vs_retic.png")
plt.close()

print(f"✓  All ten plots saved to: {out_dir}/")