import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LinearRegression
import numpy as np

# 1. Load your file
df = pd.read_csv(Path("results.csv"))

fig, ax = plt.subplots(figsize=(7,5))

# 2. Scatter points
ax.scatter(df["retic_ratio"], df["pre_ratio"], label="pre_ratio", marker="o")
ax.scatter(df["retic_ratio"], df["post_ratio"], label="post_ratio", marker="s")

# 3. Add a straight-line trend for each series
for col, style in [("pre_ratio", "k--"), ("post_ratio", "k:")]:
    # drop NaNs (where exact_result was 0)
    tmp = df[["retic_ratio", col]].dropna()
    if len(tmp) > 1:
        X = tmp["retic_ratio"].to_numpy().reshape(-1,1)
        y = tmp[col]
        reg = LinearRegression().fit(X, y)
        xfit = np.linspace(X.min(), X.max(), 100).reshape(-1,1)
        yfit = reg.predict(xfit)
        ax.plot(xfit, yfit, style, linewidth=1)

# 4. Cosmetics
ax.set_xlabel("Reticulation ratio")
ax.set_ylabel("Result / Exact result")
ax.legend(title="Series")
ax.axhline(1, color="grey", linewidth=0.8)       # ‘perfect’ line
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()
