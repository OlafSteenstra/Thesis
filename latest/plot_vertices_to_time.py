import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

df = pd.read_csv(Path("results.csv"))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 1. vertices → time
axes[0].scatter(df["vertices"], df["exact_time"], marker="o", label="exact_time")
axes[0].scatter(df["vertices"], df["algo_time"],  marker="s", label="algo_time")
axes[0].set_xlabel("Vertices")
axes[0].set_ylabel("Time (s)")
axes[0].set_title("Vertices vs computation time")
axes[0].legend()
axes[0].grid(alpha=0.3)
axes[0].set_yscale("log")          # ← log-scale time axis here

# 2. retic_ratio → time
axes[1].scatter(df["retic_ratio"], df["exact_time"], marker="o", label="exact_time")
axes[1].scatter(df["retic_ratio"], df["algo_time"],  marker="s", label="algo_time")
axes[1].set_xlabel("Reticulation ratio")
axes[1].set_ylabel("Time (s)")
axes[1].set_title("Reticulation ratio vs computation time")
axes[1].legend()
axes[1].grid(alpha=0.3)
axes[1].set_yscale("log")          # ← and here

plt.tight_layout()
plt.show()
