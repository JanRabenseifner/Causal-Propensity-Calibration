# Calibration Strategies for Robust Causal Estimation

![GitHub](https://img.shields.io/github/license/JanRabenseifner/Causal-Propensity-Calibration)
[![arXiv](https://img.shields.io/badge/arXiv-2503.17290-b31b1b.svg)](https://arxiv.org/abs/2503.17290)

This repository contains replication code for the paper **"Calibration Strategies for Robust Causal Estimation: Theoretical and Empirical Insights on Propensity Score Based Estimators"**.
## File Structure

### Evaluation Results
- `evaluate_results_drug_plots.ipynb`: Analysis and visualization for drug dataset experiments
- `evaluate_results_irm_plots.ipynb`: Interactive Regression Model (IRM) result visualization
- `evaluate_results_nonlinear_plots.ipynb`: Nonlinear scenario analysis
- `evaluate_results_unbalanced_plots.ipynb`: Unbalanced dataset evaluation
- `results_table.ipynb`: Summary tables generation
- `utils_eval.py`: Evaluation metrics and result processing


### Simulation Code
**/run-simulation/py-files**
- `sim_irm.py`: IRM DGP simulation
- `sim_drug.py`: Drug DGP simulation
- `sim_nonlinear.py`: Nonlinear DGP simulation
- `sim_unbalance.py`: Unbalanced dataset simulation
- `utils_dgps.py`: Data Generating Process (DGP) configurations
- `utils_calibration.py`: Propensity calibration methods

**/run-simulation/sh-files**
- `simulation_drug.sh`: Drug simulation bash script
- `simulation_irm.sh`: IRM simulation bash script
- `simulation_nonlinear.sh`: Nonlinear simulation bash script
- `simulation_unbalanced.sh`: Unbalanced simulation bash script

## Usage
! Note to replicators:
- Update job scheduler commands in bash scripts (SLURM/PBS/etc.) for your cluster
- Repitions are executed in parallel with different seeds:
    - Adjust --jobs parameter in parallel command to match your core limits
    - Adjust n_rep in simulation files (modify for different repetition counts)

### Local Setup
**Requirements:**
- Python 3.10+
- Jupyter Lab 3.0+
- DoubleML 0.6+
- scikit-learn 1.0+
- LightGBM 4.0+

**Installation:**
```bash
pip install -r requirements.txt
jupyter nbextension enable --py widgetsnbextension
