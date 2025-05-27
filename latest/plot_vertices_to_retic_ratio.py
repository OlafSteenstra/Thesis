import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns            # only for the violin/strip helpers
from pathlib import Path

# ── load your CSV ───────────────────────────────────────────────
df = pd.read_csv(Path("results.csv"))

# ── set up one figure ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7,4))

# 1) Violin or box summarising the distribution
sns.violinplot(
    data=df,
    x="vertices",               # treat vertices as categories
    y="retic_ratio",
    inner=None,                 # don’t draw the quartile bars – keeps it clean
    color="lightgrey",
    ax=ax,
)

# 2) Raw data on top (jittered for visibility)
sns.stripplot(
    data=df,
    x="vertices",
    y="retic_ratio",
    hue="method",               # optional: colour by algorithm if you have several
    dodge=True,                 # keeps overlapping methods separated
    jitter=0.2,                 # horizontal jitter
    alpha=0.6,
    size=4,
    ax=ax,
)

# ── tidy up ─────────────────────────────────────────────────────
ax.set_title("Distribution of reticulation ratio by graph size")
ax.set_xlabel("Vertices")
ax.set_ylabel("Reticulation ratio")
ax.grid(axis="y", alpha=.3)
ax.legend(title="method")        # remove if you passed no hue

plt.tight_layout()
plt.show()
