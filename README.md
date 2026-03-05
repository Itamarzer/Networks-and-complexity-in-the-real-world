# The Stochastic Engine: Optimization and Structural Order in Dynamic Brain Networks

This repository contains the Python analysis pipeline, statistical testing, and visualization code for the project **"The Stochastic Engine: Optimization and Structural Order in Dynamic Brain Networks."** This project challenges the traditional localizationist view of the human brain as a static hierarchy of fixed functions. By modeling resting-state fMRI (rs-fMRI) data as a time-varying complex network, we demonstrate that the brain's apparent "noise" is actually the driving force of a highly optimized computational engine. The codebase empirically traces the brain's operational trajectory from micro-scale stochasticity to macro-scale thermodynamic efficiency.

## Project Overview

The core analytical pipeline is divided into three primary hypotheses, demonstrating the concept of **Intelligent Stochasticity**:

* **Hypothesis I: The Fluid Topology (Stochastic Exploration)** Analyzes the dynamic reconfiguration of the brain using sliding-window functional connectivity and Spectral Clustering. It proves non-stationarity by generating phase-randomized surrogate networks, showing the brain's temporal flexibility significantly exceeds random mechanical noise.
  
* **Hypothesis II: The Tuning Cycle (Topological Optimization)**
  Evaluates the ongoing trade-off between network Integration (Dynamic Global Efficiency, **Eg**) and Segregation (Dynamic Modularity, **Q**). It maps the brain's dynamic topology against a Pareto frontier and Erdős-Rényi random graphs, demonstrating that the brain selectively achieves critical "Super-States."
  
* **Hypothesis III: Thermodynamic Optimization & Network Control**
  Models the brain's temporal sequence as a Discrete-Time Markov Chain. Applying the Barzel framework from Network Control Theory (calculating **x_eff** and **β_eff**), the code proves that the brain minimizes thermodynamic cost. It establishes a significant negative correlation between Network Entropy (energy cost) and State Persistence (dwell time).

## Dataset & Preprocessing

* **Data Source:** ADHD-200 resting-state fMRI dataset (fetched dynamically via `nilearn`).
* **Atlas:** Multi-Subject Dictionary Learning (MSDL) atlas, providing 39 functional regions of interest (ROIs).
* **Pipeline:** * Confound regression (head motion, white matter, CSF).
  * Signal standardization.
  * Dynamic network construction using a sliding window approach (e.g., 15-30 timepoints window, 1 TR step size).
  * Matrix sparsification via proportional thresholding.

## Repository Structure

The repository is organized by hypothesis. Each folder contains the specific script required to run that part of the analysis.

```text
├── hypothesis_1/
│   └── hypothesis1_stochastic_engine.py
├── hypothesis_2/
│   └── hypothesis2_tuning_cycle.py
├── hypothesis_3/
│   └── hypothesis3_thermodynamics.py
├── requirements.txt
└── README.md


## Installation & Setup

It is highly recommended to run this project inside an isolated Python virtual environment to avoid dependency conflicts.

**1. Clone the repository:**

```bash
git clone <your-repository-url>
cd <repository-name>

```

**2. Create a virtual environment:**

* **On macOS and Linux:**
```bash
python3 -m venv venv
source venv/bin/activate

```


* **On Windows:**
```cmd
python -m venv venv
venv\Scripts\activate

```



**3. Install the dependencies:**
Ensure your virtual environment is activated, then run:

```bash
pip install -r requirements.txt

```

*(This will install necessary packages including `numpy`, `scipy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `networkx`, `nilearn`, and `nibabel`.)*

## How to Run

Navigate into the respective hypothesis folder and run the python script. Each script will automatically download the required fMRI data (if not already cached), perform the analysis, and generate a dedicated output directory containing the results.

**Run Hypothesis I (Fluid Topology):**

```bash
cd hypothesis_1
python hypothesis1_stochastic_engine.py
cd ..

```

**Run Hypothesis II (The Tuning Cycle):**

```bash
cd hypothesis_2
python hypothesis2_tuning_cycle.py
cd ..

```

**Run Hypothesis III (Markovian Dynamics & Thermodynamics):**

```bash
cd hypothesis_3
python hypothesis3_thermodynamics.py
cd ..

```

## Outputs

Executing each script generates an output directory (e.g., `hypothesis_3/hypothesis3_partb_final_publication/`) containing:

* `/data/`: Saved numpy arrays (`.npy`), extracted time series, and `.csv` reports.
* `/statistics_txt/`: Comprehensive statistical reports detailing p-values, correlations, Mann-Whitney U results, and effect sizes.
* `/plots_pdf/`: High-resolution, publication-ready figures, including dynamic cluster assignments, phase space distributions, surrogate comparisons, and thermodynamic optimization metrics.

## Contact

For questions, discussions, or collaboration inquiries regarding this research or the codebase, please reach out:

**Itamar Zernitsky**

* **Email:** itamar.zernitsky@gmail.com
* **Institution:** Bar Ilan University, Department of Mathematics

```

```
