import pandas as pd
import statsmodels.formula.api as smf

def perform_regression(file_path='results.csv'):
    """
    Performs a multiple linear regression analysis on the provided CSV file.

    Predicting variables: 'vertices', 'retic_ratio', and 'vertices * retic_ratio'
    Variable to be predicted: 'exact_time'

    Args:
        file_path (str): The path to the CSV file (default: 'results.csv').
    """

    # --- 1. Load the Data ---
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
        print("Creating a dummy 'results.csv' for demonstration purposes.")
        print("Please replace this with your actual data.\n")

        # Create a dummy DataFrame for demonstration if file not found
        data = {
            'vertices': [10, 20, 15, 25, 30, 12, 22, 18, 28, 35],
            'retic_ratio': [0.5, 0.7, 0.6, 0.8, 0.9, 0.55, 0.75, 0.65, 0.85, 0.95],
            'exact_time': [1.2, 3.5, 2.0, 5.0, 7.5, 1.5, 4.0, 2.8, 6.0, 8.8]
        }
        df = pd.DataFrame(data)
        df.to_csv(file_path, index=False)
        print("Dummy 'results.csv' created. Please run the script again with your real data.")
        return # Exit the function as dummy data was just created

    # --- 2. Check for Required Columns ---
    required_columns = ['vertices', 'retic_ratio', 'exact_time']
    if not all(col in df.columns for col in required_columns):
        missing = [col for col in required_columns if col not in df.columns]
        print(f"\nError: Missing required columns in '{file_path}'. Missing: {missing}")
        print("Please ensure your CSV has 'vertices', 'retic_ratio', and 'exact_time' columns.")
        return

    formula_parts = [
        "exact_time ~",
        "retic_ratio:vertices",  # Interaction term for retic_ratio and vertices
        "+ I(vertices**2):I(retic_ratio**2)"
        "+ I(vertices**3):I(retic_ratio**3)"
    ]
    formula = " ".join(formula_parts)

    print(f"\nRegression Model Formula: {formula}")

    # --- 4. Perform the Regression Analysis ---
    try:
        model = smf.ols(formula=formula, data=df).fit()

        # --- 5. Display the Results ---
        print("\n--- Regression Analysis Results ---")
        print(model.summary())
        print(f"\nModel Coefficients:\n{model.params}")
        print(f"\nR-squared: {model.rsquared:.4f}")

        with open("regression_results.tex", "w") as f:
            f.write(model.summary().as_latex())
    except Exception as e:
        print(f"\nAn error occurred during regression analysis: {e}")

# --- Run the script ---
if __name__ == "__main__":
    perform_regression('results.csv')