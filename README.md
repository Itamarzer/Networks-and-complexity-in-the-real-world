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
