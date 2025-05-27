import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.linear_model import LinearRegression
import numpy as np

# ── load your data ──────────────────────────────────────────────
df = pd.read_csv(Path("results.csv"))

# ── figure set-up ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6.5, 4.5))

sizes   = sorted(df["vertices"].unique())
palette = sns.color_palette("tab10", len(sizes))
size2c  = dict(zip(sizes, palette))

# ── scatter + per-group trendlines ──────────────────────────────
for v, colour in size2c.items():
    subset = df[df["vertices"] == v]

    # scatter points
    ax.scatter(
        subset["retic_ratio"],
        subset["post_ratio"],
        label=f"{v} vertices",
        color=colour,
        alpha=0.65,
        edgecolor="none",
        s=35,
    )

    # linear trend (use LOWESS if you prefer non-parametric)
    if len(subset) > 1:                       # need ≥2 points to fit
        X = subset["retic_ratio"].to_numpy().reshape(-1, 1)
        y = subset["post_ratio"].to_numpy()
        reg = LinearRegression().fit(X, y)
        xfit = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        ax.plot(
            xfit,
            reg.predict(xfit),
            color=colour,
            linewidth=1.5,
        )

# ── cosmetics ──────────────────────────────────────────────────
ax.set_xlabel("Reticulation ratio")
ax.set_ylabel("Post-ratio (post_result / exact_result)")
ax.set_title("Post-ratio vs. reticulation, with trend by graph size")
ax.axhline(1, color="grey", linestyle="--", linewidth=0.8)
ax.legend(title="Graph size")
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()
