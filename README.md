# Phylogenetic Network Parsimony Score Approximation

This project implements an approximation algorithm for computing the softwired parsimony score on phylogenetic networks. It provides functionality to run the approximation, compare its performance against an external exact solver (MPNet), visualize the algorithm's steps, generate random networks for testing, and analyze the results through plots and regression.

## Features

*   **Approximation Algorithm:** Implements a specific algorithm for calculating the softwired parsimony score on arbitrary phylogenetic networks.
*   **Exact Solver Comparison:** Integrates with the external Java-based `MPNet` tool to obtain exact parsimony scores for benchmarking.
*   **Visualization:** Step-by-step visualization of the algorithm's processing and state changes, output as animated GIFs.
*   **Network Generation:** Includes functionality to generate random phylogenetic networks with varying sizes and reticulation ratios for robust testing.
*   **Data Analysis & Plotting:** Scripts to process benchmark results (`results.csv`) and generate insightful plots using `matplotlib` and `seaborn`.
*   **Regression Analysis:** Statistical analysis using `statsmodels` to explore relationships between network characteristics and performance metrics.

## Prerequisites

Before setting up the Python environment, ensure you have the following external software installed:

1.  **Java Development Kit (JDK):** Required to run the `MPNet` exact solver.
2.  **IBM ILOG CPLEX Optimization Studio:** The `cplex` component is necessary for the `MPNet` solver to function. The current code assumes CPLEX Studio 22.1.2 (or compatible).
3.  **MPNet Java Executable:** You need the compiled `MPNet.jar` or `MPNet.class` files from the `MPNet` project. This project assumes you have the `MPNet.java` source and have compiled it into a class file.
4.  **Graphviz:** Highly recommended for `networkx` to generate visually appealing and efficient network layouts for visualization. Install it via your system's package manager (e.g., `sudo apt-get install graphviz` on Ubuntu, `brew install graphviz` on macOS).

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/your-repo-name.git # Replace with your actual repo
    cd your-repo-name
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate # On Windows: .\venv\Scripts\activate
    ```

3.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Configuration

Before running, you **must** configure the paths for CPLEX and your working directory in `MPNet.py`:

Open `MPNet.py` and modify the following lines to reflect your system's paths:

```python
CPLEX_HOME = Path(r"C:\Program Files\IBM\ILOG\CPLEX_Studio2212") # <-- Set this to your CPLEX installation path
work_dir   = Path(r"C:\Users\jantj\Documents\mpnet")              # <-- Set this to a working directory where MPNet can read/write files
```

Additionally, ensure that the compiled `MPNet` Java class file (`MPNet.class`) is located within the `work_dir` specified above, or adjust the `java_cmd` in `MPNet.py` accordingly. If you have the Java source for MPNet (e.g., `MPNet.java`), you can compile it:
```bash
javac MPNet.java
```
Make sure the `MPNet.class` file is in the `work_dir`.

## Usage

### Running the Approximation Algorithm (Ad-hoc)

You can run the approximation algorithm directly on a network:

```python
# Example of running the algorithm from TestCases.py
from Code.TestCases import TestCases

test = TestCases()
network, char = test.test_case_1() # Or any other test_case or a network you create

pre_score, post_score, assignment = network.run_approximation(char)
print(f"Pre-improvement score: {pre_score}")
print(f"Post-improvement score: {post_score}")
print(f"Final assignment: {assignment}")
```

### Generating Test Data & Benchmarking

The `Testing.py` script automates the process of generating random networks, running both the approximation and exact solvers, and saving the results to `results.csv`.

```bash
python Testing.py
```
This script will run indefinitely, generating and testing random networks until stopped.

### Plotting Results

After generating `results.csv` using `Testing.py`, you can generate various plots to analyze the performance:

```bash
python Plots.py
```
This will create a `plots/` directory containing PNG images of the generated plots.

### Regression Analysis

To perform a statistical regression on the data in `results.csv`:

```bash
python Regression.py
```
The script will print a summary of the regression model to the console and also save a LaTeX formatted summary to `regression_results.tex`.

### Visualization

To run the approximation algorithm with step-by-step visualization (outputs a GIF):

```python
# Example: Visualize test_case_1
from Code.TestCases import TestCases

test = TestCases()
network_viz, char = test.test_case_1(viz=True) # Pass viz=True to get a visualization-enabled network

# Run the algorithm; output_video and fps are optional
network_viz.run_approximation(char, output_video="test_case_1_viz.gif", fps=1, keep_frames=False)
```
This will generate `test_case_1_viz.gif` in your project root, showing the algorithm's steps.

## Project Structure

*   `MPNet.py`: Python interface to the external Java-based `MPNet` exact parsimony solver.
*   `PhylogeneticNetwork.py`: Core implementation of the phylogenetic network data structure and the non-visual approximation algorithm.
*   `PhylogeneticNetwork_Viz.py`: Extends `PhylogeneticNetwork.py` to include visualization capabilities for algorithm steps.
*   `Helper.py`: Contains utility functions for parsing Newick strings, reading character data, exporting network data for `MPNet`, and comparing results.
*   `TestCases.py`: Defines various hardcoded and random phylogenetic network test cases, along with their character data.
*   `Testing.py`: The main script for running automated benchmarks, generating random networks, and recording performance data.
*   `Plots.py`: Script for generating analytical plots from the `results.csv` data.
*   `Regression.py`: Script for performing statistical regression analysis on the benchmark results.
*   `Vertex.py`: Defines the `Vertex` class used as nodes in the phylogenetic network.
*   `requirements.txt`: Lists all Python package dependencies.
*   `results.csv` (Generated): Stores the outcomes of benchmark tests.
*   `plots/` (Generated): Directory where generated plots are saved.
*   `_algo_viz_frames/` (Generated, temporary): Directory used by `PhylogeneticNetwork_Viz.py` to store individual frames before compiling into a GIF/video.