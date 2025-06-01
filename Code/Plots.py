import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

csv_file = "results.csv"
out_dir  = "plots"
os.makedirs(out_dir, exist_ok=True)

"""Creates a wide array of plots based on experimental results from a CSV file."""
df = pd.read_csv(csv_file)

df["speedup"]      = df["exact_time"] / df["algo_time"]
df["approx_ratio"] = df["post_ratio"]
df["solved"]       = ~df["exact_time"].isna()
df["retics"]       = df["retic_ratio"] * df["vertices"]
df["diff"]         = df["pre_result"] - df["post_result"]
df["diff_in_ratio"] = df["pre_ratio"] - df["post_ratio"]
df["diff_ratio"]   = df["diff"] / df["exact_result"]

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

sns.set_style("whitegrid")
plt.rcParams["savefig.dpi"] = 300

plt.figure(figsize=(6, 4))
sns.scatterplot(data=df, x="vertices", y="diff")
plt.axhline(0, ls="--", c="grey", lw=.8)
plt.xlabel("Internal vertices")
plt.ylabel("Difference in SPS")
plt.title("Difference between pre- and post-improvement SPS")
plt.tight_layout()
plt.savefig(f"{out_dir}/diff_vs_size.png")
plt.close()

plt.figure(figsize=(6, 4))
sns.scatterplot(data=df, x="retics", y="diff")
plt.axhline(0, ls="--", c="grey", lw=.8)
plt.xlabel("Reticulation vertices")
plt.ylabel("Difference in SPS")
plt.title("Difference between pre- and post-improvement SPS")
plt.tight_layout()
plt.savefig(f"{out_dir}/diff_vs_retics.png")
plt.close()

plt.figure(figsize=(6, 4))
sns.scatterplot(data=df, x="retic_ratio", y="diff")
plt.axhline(0, ls="--", c="grey", lw=.8)
plt.xlabel("Reticulation vertices")
plt.ylabel("Difference in SPS")
plt.title("Difference between pre- and post-improvement SPS")
plt.tight_layout()
plt.savefig(f"{out_dir}/diff_vs_retic.png")
plt.close()

plt.figure(figsize=(6, 4))
sns.scatterplot(data=df, x="vertices", y="diff_ratio")
plt.axhline(0, ls="--", c="grey", lw=.8)
plt.xlabel("Internal vertices")
plt.ylabel("Difference in approximation ratio")
plt.title("Difference between pre- and post-improvement approximation ratio versus internal vertices")
plt.tight_layout()
plt.savefig(f"{out_dir}/diff_r_vs_size.png")
plt.close()

plt.figure(figsize=(6, 4))
sns.scatterplot(data=df, x="retics", y="diff_ratio")
plt.axhline(0, ls="--", c="grey", lw=.8)
plt.xlabel("Reticulation vertices")
plt.ylabel("Difference in approximation ratio")
plt.title("Difference between pre- and post-improvement approximation ratio versus reticulation vertices")
plt.tight_layout()
plt.savefig(f"{out_dir}/diff_r_vs_retics.png")
plt.close()

plt.figure(figsize=(6, 4))
sns.scatterplot(data=df, x="retic_ratio", y="diff_ratio")
plt.axhline(0, ls="--", c="grey", lw=.8)
plt.xlabel("Reticulation ratio")
plt.ylabel("Difference in approximation ratio")
plt.title("Difference between pre- and post-improvement approximation ratio versus reticulation ratio")
plt.tight_layout()
plt.savefig(f"{out_dir}/diff_r_vs_retic.png")
plt.close()

plt.figure(figsize=(6, 4))
sns.scatterplot(data=df, x="retic_ratio_goal", y="retic_ratio")
plt.xlabel("Reticulation ratio goal")
plt.ylabel("Reticulation ratio achieved")
plt.title("Reticulation ratio goal vs achieved")
plt.tight_layout()
plt.savefig(f"{out_dir}/retic_goal_vs_achieved.png")
plt.close()

plt.figure(figsize=(6, 4))
sns.scatterplot(data=df, x="vertices", y="exact_time",  label="Exact (MPNet)", marker="o")
sns.scatterplot(data=df, x="vertices", y="algo_time",   label="Approx.",       marker="s")
plt.yscale("log")
plt.xlabel("Internal vertices")
plt.ylabel("Runtime (s, log)")
plt.title("Runtime vs graph size")
plt.tight_layout()
plt.legend()
plt.savefig(f"{out_dir}/runtime_vs_size.png")
plt.close()

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
plt.xlabel("Reticulation vertices")
plt.ylabel("Runtime (s, log)")
plt.title("Runtime vs reticulation")
plt.tight_layout()
plt.legend()
plt.savefig(f"{out_dir}/runtime_vs_retics.png")
plt.close()

for xcol, name in [("vertices", "size"), ("retic_ratio", "retic"), ("retics", "reticulation vertices")]:	
    plt.figure(figsize=(6, 4))
    sns.scatterplot(data=df, x=xcol, y="speedup", alpha=0.7)
    sns.regplot(
        data=df,
        x=xcol,
        y="speedup",
        scatter=False,
        lowess=True,
        color="black",
        line_kws={"lw": 1.2},
    )

    if xcol == "vertices":
        plt.xlabel("Internal vertices")
    elif xcol == "retic_ratio":
        plt.xlabel("Reticulation ratio")
    else:
        plt.xlabel("Reticulation vertices")

    plt.yscale("log")
    plt.ylabel("Speed-up  (exact / approx, log)")
    plt.title(f"Algorithm speed-up vs {name}")

    plt.tight_layout()
    plt.savefig(f"{out_dir}/speedup_vs_{name}.png")
    plt.close()

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
sns.histplot(df["pre_ratio"].dropna(), bins=30, ax=axes[0])
axes[0].set_xlabel("Approximation ratio")
axes[0].set_title("Distribution of pre-improvement approximation ratios")

sns.histplot(df["approx_ratio"].dropna(), bins=30, ax=axes[1])
axes[1].set_xlabel("Approximation ratio")
axes[1].set_title("Distribution of post-improvement approximation ratios")
plt.tight_layout()
plt.savefig(f"{out_dir}/combined_histograms.png")
plt.close(fig)

plot_df = df[['pre_ratio', 'approx_ratio']].copy()
plot_df.columns = ['Pre-Improvement', 'Post-Improvement']
melted_df = plot_df.melt(var_name='Improvement Stage', value_name='Approximation Ratio')
melted_df.dropna(inplace=True)
plt.figure(figsize=(8, 5))
sns.histplot(
    data=melted_df,
    x="Approximation Ratio",
    hue="Improvement Stage",
    bins=30,
    kde=False,
    alpha=0.6,
    common_bins=True,
    common_norm=False
)
plt.xlabel("Approximation ratio")
plt.ylabel("Count")
plt.title("Distribution of Approximation Ratios (Pre- vs Post-Improvement)")
plt.tight_layout()
plt.savefig(f"{out_dir}/overlaid_histograms.png")
plt.close()

plt.figure(figsize=(6, 4))
sns.scatterplot(data=df, x="pre_ratio", y="approx_ratio", alpha=.7)
plt.xlabel("Pre-improvement approximation ratio")
plt.ylabel("Post-improvement approximation ratio")
plt.title("Pre- vs post-improvement approximation ratios")
plt.savefig(f"{out_dir}/pre_vs_post_approx_ratio.png")
plt.close()

plt.figure(figsize=(6, 4))
sns.scatterplot(data=df, x="pre_ratio", y="diff_in_ratio", alpha=.7)
plt.xlabel("Pre-improvement approximation ratio")
plt.ylabel("Improvement in approximation ratio")
plt.title("Improvement in approximation ratios after algorithmic improvement")
plt.savefig(f"{out_dir}/pre_vs_diff_approx_ratio.png")
plt.close()

plt.figure(figsize=(6, 4))
sns.ecdfplot(df["approx_ratio"].dropna())
plt.xlabel("Approximation ratio")
plt.ylabel("Cumulative fraction")
plt.title("ECDF of approximation ratios")
plt.tight_layout()
plt.savefig(f"{out_dir}/ratio_ecdf.png")
plt.close()

plt.figure(figsize=(6, 4))
sns.scatterplot(data=df, x="retic_ratio", y="approx_ratio", hue="vert_bin", alpha=.7)
plt.axhline(1, ls="--", c="grey", lw=.8)
plt.xlabel("Reticulation ratio")
plt.ylabel("Approximation ratio")
plt.title("Reticulation ratio vs approximation ratio")
plt.legend(title="Internal vertices", loc="upper left")
plt.tight_layout()
plt.savefig(f"{out_dir}/ratio_vs_retic.png")
plt.close()

plt.figure(figsize=(6, 4))
sns.scatterplot(data=df, x="retic_ratio", y="pre_ratio", hue="vert_bin", alpha=.7)
plt.axhline(1, ls="--", c="grey", lw=.8)
plt.xlabel("Reticulation ratio")
plt.ylabel("Approximation ratio")
plt.title("Reticulation ratio vs approximation ratio")
plt.legend(title="Internal vertices", loc="upper left")
plt.tight_layout()
plt.savefig(f"{out_dir}/pre_ratio_vs_retic.png")
plt.close()

print(f"All plots saved.")