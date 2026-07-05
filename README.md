# Code for "Error Bounds for Multi-Task Gaussian Processes with Uncertain Task Correlations"

This repository contains the source code and supplementary material for the paper:

> Jannis O. Lübsen and Annika Eichler, "Error Bounds for Multi-Task Gaussian Processes with Uncertain Task Correlations"

The code allows for the reproduction of all tables and figures presented in the manuscript.

# Error Bounds for Multi-Task Gaussian Processes with Uncertain Task Correlations

This repository contains the code required to reproduce all tables and figures presented in the manuscript: **"Safe Bayesian Optimization for Uncertain Correlations Matrices in Linear Models of Coregionalization."**

## Prerequisites

The codebase was developed and tested in the following environment:
* **OS:** Ubuntu 24.04.2 LTS
* **Language:** Python 3.12.8

## Installation

To set up the environment, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone 
    ```

2.  **Navigate to the project directory:**
    ```bash
    cd ./2026-code-Error-Bounds-for-Multi-Task-Gaussian-Processes-with-Uncertain-Task-Correlations
    ```

3.  **Install dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```

## Reproducing Results

### 1. Running the Optimization Algorithms

**Multitask Bayesian Optimization:**
To run the multitask algorithm, execute `python3 run_samsbo.py LbSync <disturbance>` script. You can specify the disturbance scaling by substituing <disturbance> with a float, e.g. {0.1,0.15,0.25} as used in the paper 

```bash
# Run with 0.1 disturbance factor
python3 run_samsbo.py LbSync 0.1
```

**Single Task Bayesian Optimization:**
For the single-task case, execute the `run_comparison.py` script.

```bash
python run_comparison.py LbSync st_constraints
```

### 2. Generating Plots ###

To generate the plots used in the manuscript:
1. Navigate to the `plot_scripts` directory
2. Open and run the Jupyter Notebook `generate_plots.ipynb`

