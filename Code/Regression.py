import pandas as pd
import statsmodels.formula.api as smf

def perform_regression(file_path='results.csv'):
    """Performs a regression analysis on experimental results from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        df["speedup"] = df["exact_time"] / df["algo_time"]
        df["diff"] =  df["pre_ratio"] - df["post_ratio"]
        print(f"Successfully loaded data from '{file_path}'")
        print("\nFirst 5 rows of the data:")
        print(df.head())
        print(f"\nDataFrame shape: {df.shape}")
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return

    required_columns = ['vertices', 'retic_ratio', 'exact_time']
    if not all(col in df.columns for col in required_columns):
        missing = [col for col in required_columns if col not in df.columns]
        print(f"\nError: Missing required columns in '{file_path}'. Missing: {missing}")
        print("Please ensure your CSV has 'vertices', 'retic_ratio', and 'exact_time' columns.")
        return

    formula_parts = [
        "exact_time ~",
        "retic_ratio:vertices",
        "+ I(vertices**2):I(retic_ratio**2)"
        "+ I(vertices**3):I(retic_ratio**3)"
    ]
    formula = " ".join(formula_parts)

    print(f"\nRegression Model Formula: {formula}")

    try:
        model = smf.ols(formula=formula, data=df).fit()

        print("\n--- Regression Analysis Results ---")
        print(model.summary())
        print(f"\nModel Coefficients:\n{model.params}")
        print(f"\nR-squared: {model.rsquared:.4f}")

        with open("regression_results.tex", "w") as f:
            f.write(model.summary().as_latex())
    except Exception as e:
        print(f"\nAn error occurred during regression analysis: {e}")

perform_regression('results.csv')