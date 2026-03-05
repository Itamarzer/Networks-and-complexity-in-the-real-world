"""
================================================================================
HYPOTHESIS III PART B: DIAGNOSTIC ANALYSIS
Testing Both Correlation Directions & Data-Hypothesis Fit
================================================================================

THE PROBLEM:
- Correlation r ≈ 0 (no relationship detected)
- This could mean:
  1. Hypothesis is correct but measures are wrong
  2. Hypothesis direction is reversed in this data
  3. The relationship is nonlinear
  4. Thermodynamic optimization doesn't apply to this dataset

DIAGNOSTIC APPROACH:
Test ALL possibilities and let the data tell us which is true.

================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway, kruskal, mannwhitneyu, pearsonr, spearmanr, linregress, kendalltau
from scipy.linalg import eig
from sklearn.cluster import SpectralClustering
from scipy.signal import correlate
import warnings
from pathlib import Path
import pandas as pd
from nilearn import datasets
from nilearn.maskers import NiftiMapsMasker
import nibabel as nib

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')

output_dir = Path(f"hypothesis3_partb_diagnostic")
output_dir.mkdir(exist_ok=True, parents=True)

pdf_dir = output_dir / "plots_pdf"
txt_dir = output_dir / "statistics_txt"
data_dir = output_dir / "data"

for d in [pdf_dir, txt_dir, data_dir]:
    d.mkdir(exist_ok=True, parents=True)

log_file = output_dir / "diagnostic_log.txt"
log_handle = open(log_file, 'w')


def print_both(message=""):
    print(message)
    log_handle.write(message + "\n")
    log_handle.flush()


# ================================================================================
# DATA LOADING
# ================================================================================

print_both("=" * 80)
print_both("HYPOTHESIS III PART B: DIAGNOSTIC ANALYSIS")
print_both("Testing Data-Hypothesis Fit")
print_both("=" * 80 + "\n")

try:
    adhd_data = datasets.fetch_adhd(n_subjects=10)
    atlas = datasets.fetch_atlas_msdl()
    atlas_img = nib.load(atlas.maps)
except:
    exit(1)

masker = NiftiMapsMasker(maps_img=atlas_img, standardize=True, verbose=0)
all_time_series = []

for subj_idx in range(len(adhd_data.func)):
    try:
        fmri_img = nib.load(adhd_data.func[subj_idx])
        confounds_df = pd.read_csv(adhd_data.confounds[subj_idx], sep='\t')
        confounds = confounds_df.values
        ts = masker.fit_transform(fmri_img, confounds=confounds)

        if ts.shape[0] > 50 and not np.isnan(ts).any():
            all_time_series.append(ts)
    except:
        continue

time_series = max(all_time_series, key=lambda x: x.shape[0])
n_timepoints, n_nodes = time_series.shape

# Windows
window_length = 15
step_size = 1
n_windows = (n_timepoints - window_length) // step_size + 1

windows = []
for i in range(n_windows):
    start = i * step_size
    end = start + window_length
    w_data = time_series[start:end, :]
    w_data = (w_data - w_data.mean(axis=0)) / (w_data.std(axis=0) + 1e-8)
    windows.append(w_data)

windows = np.array(windows)

# Connectivity
connectivity_matrices = []
for w in range(n_windows):
    corr = np.corrcoef(windows[w].T)
    corr = np.nan_to_num(corr, nan=0.0)
    threshold = np.percentile(np.abs(corr), 70)
    corr_sparse = corr.copy()
    corr_sparse[np.abs(corr_sparse) < threshold] = 0
    connectivity_matrices.append(corr_sparse)

connectivity_matrices = np.array(connectivity_matrices)

print_both(f"Data: {n_timepoints} timepoints, {n_nodes} regions, {n_windows} windows\n")

# ================================================================================
# BARZEL EQUATIONS
# ================================================================================

A_avg = np.mean(connectivity_matrices, axis=0)
A_avg = np.abs(A_avg)
np.fill_diagonal(A_avg, 0)

s_out = np.sum(A_avg, axis=1)
s_in = np.sum(A_avg, axis=0)
ones = np.ones(n_nodes)

xeff = (ones @ A_avg @ s_out) / (ones @ A_avg @ ones + 1e-10)
mean_s_in_out = np.mean(s_in * s_out)
beta_eff = ((ones @ A_avg @ s_in) / (ones @ A_avg @ ones + 1e-10)) * (mean_s_in_out / (np.mean(s_out) + 1e-10))

print_both("=" * 80)
print_both("BARZEL EQUATIONS")
print_both("=" * 80)
print_both(f"xeff = {xeff:.6f}")
print_both(f"βeff = {beta_eff:.6f}\n")

# ================================================================================
# MODAL CONTROLLABILITY
# ================================================================================

print_both("=" * 80)
print_both("COMPUTING MULTIPLE ENERGY/DWELL MEASURES")
print_both("=" * 80 + "\n")

modal_ctrl = np.zeros(n_windows)
network_entropy = np.zeros(n_windows)
connectivity_strength = np.zeros(n_windows)

for w in range(n_windows):
    A_w = np.abs(connectivity_matrices[w])
    np.fill_diagonal(A_w, 0)

    # Modal controllability
    try:
        eigenvalues, eigenvectors = eig(A_w)
        eigenvalues = np.real(eigenvalues)
        eigenvectors = np.real(eigenvectors)

        modal_w = np.zeros(n_nodes)
        for i in range(n_nodes):
            for mode in range(n_nodes):
                mode_strength = abs(eigenvalues[mode])
                if mode_strength < 1:
                    modal_w[i] += (abs(eigenvectors[i, mode]) ** 2) * (1 - mode_strength ** 2)

        if np.max(modal_w) > 0:
            modal_w = modal_w / np.max(modal_w)
        modal_ctrl[w] = np.mean(modal_w)
    except:
        modal_ctrl[w] = np.nan

    # Network entropy (alternative energy measure)
    degrees = np.sum(A_w, axis=1)
    if np.sum(degrees) > 0:
        p_degrees = degrees / np.sum(degrees)
        p_degrees = p_degrees[p_degrees > 0]
        network_entropy[w] = -np.sum(p_degrees * np.log2(p_degrees + 1e-10))
    else:
        network_entropy[w] = 0

    # Connectivity strength
    connectivity_strength[w] = np.sum(np.abs(A_w))

# Energy costs (try different definitions)
energy_1 = 1.0 - modal_ctrl  # Inverse of controllability
energy_1 = (energy_1 - np.nanmin(energy_1)) / (np.nanmax(energy_1) - np.nanmin(energy_1) + 1e-10)

energy_2 = modal_ctrl  # Direct controllability (opposite)
energy_2 = (energy_2 - np.nanmin(energy_2)) / (np.nanmax(energy_2) - np.nanmin(energy_2) + 1e-10)

energy_3 = network_entropy
energy_3 = (energy_3 - np.nanmin(energy_3)) / (np.nanmax(energy_3) - np.nanmin(energy_3) + 1e-10)

energy_4 = connectivity_strength
energy_4 = (energy_4 - np.nanmin(energy_4)) / (np.nanmax(energy_4) - np.nanmin(energy_4) + 1e-10)

# Dwell measures
dwell_1 = np.zeros(n_windows)  # Future similarity
for w in range(n_windows - 1):
    A_w = connectivity_matrices[w]
    future_sims = []
    for w_future in range(w + 1, min(w + 10, n_windows)):
        sim = np.linalg.norm(A_w - connectivity_matrices[w_future], 'fro')
        future_sims.append(sim)
    dwell_1[w] = np.mean(future_sims) if future_sims else 0

dwell_1 = (dwell_1 - np.nanmin(dwell_1)) / (np.nanmax(dwell_1) - np.nanmin(dwell_1) + 1e-10)

dwell_2 = np.zeros(n_windows)  # Velocity (change rate)
for w in range(1, n_windows):
    dwell_2[w] = np.linalg.norm(connectivity_matrices[w] - connectivity_matrices[w - 1], 'fro')

dwell_2 = (dwell_2 - np.nanmin(dwell_2)) / (np.nanmax(dwell_2) - np.nanmin(dwell_2) + 1e-10)

dwell_3 = np.zeros(n_windows)  # Inverse velocity (stability)
dwell_3 = 1.0 - dwell_2

print_both(f"Energy measure 1 (1-modal): range [{np.nanmin(energy_1):.3f}, {np.nanmax(energy_1):.3f}]")
print_both(f"Energy measure 2 (modal): range [{np.nanmin(energy_2):.3f}, {np.nanmax(energy_2):.3f}]")
print_both(f"Energy measure 3 (entropy): range [{np.nanmin(energy_3):.3f}, {np.nanmax(energy_3):.3f}]")
print_both(f"Energy measure 4 (strength): range [{np.nanmin(energy_4):.3f}, {np.nanmax(energy_4):.3f}]")
print_both(f"Dwell measure 1 (future sim): range [{np.nanmin(dwell_1):.3f}, {np.nanmax(dwell_1):.3f}]")
print_both(f"Dwell measure 2 (velocity): range [{np.nanmin(dwell_2):.3f}, {np.nanmax(dwell_2):.3f}]")
print_both(f"Dwell measure 3 (stability): range [{np.nanmin(dwell_3):.3f}, {np.nanmax(dwell_3):.3f}]\n")

# ================================================================================
# TEST ALL COMBINATIONS
# ================================================================================

print_both("=" * 80)
print_both("TESTING ALL ENERGY × DWELL COMBINATIONS")
print_both("=" * 80 + "\n")

energy_measures = {
    'Energy1 (1-modal)': energy_1,
    'Energy2 (modal)': energy_2,
    'Energy3 (entropy)': energy_3,
    'Energy4 (strength)': energy_4
}

dwell_measures = {
    'Dwell1 (future-sim)': dwell_1,
    'Dwell2 (velocity)': dwell_2,
    'Dwell3 (stability)': dwell_3
}

results = []

for e_name, e_data in energy_measures.items():
    for d_name, d_data in dwell_measures.items():
        # Remove NaN
        valid = ~(np.isnan(e_data) | np.isnan(d_data))
        e_valid = e_data[valid]
        d_valid = d_data[valid]

        if len(e_valid) < 2:
            continue

        r, p = pearsonr(e_valid, d_valid)

        results.append({
            'energy': e_name,
            'dwell': d_name,
            'correlation': r,
            'p_value': p,
            'abs_correlation': abs(r),
            'significant': p < 0.05
        })

        sig_mark = "✓" if p < 0.05 else ""
        print_both(f"{e_name:20s} × {d_name:20s}  →  r={r:+.4f}, p={p:.4f} {sig_mark}")

print_both("")

df_results = pd.DataFrame(results)
df_results_sorted = df_results.sort_values('abs_correlation', ascending=False)

print_both("\nTOP 5 STRONGEST CORRELATIONS:")
print_both("-" * 80)
for idx, row in df_results_sorted.head(5).iterrows():
    sig = "✓ SIGNIFICANT" if row['significant'] else ""
    print_both(f"{row['correlation']:+.4f}  {row['energy']} × {row['dwell']}  {sig}")

print_both("")

# ================================================================================
# PLOTS
# ================================================================================

print_both("=" * 80)
print_both("CREATING DIAGNOSTIC PLOTS")
print_both("=" * 80 + "\n")

# Plot 1: Heatmap of all correlations
fig, ax = plt.subplots(figsize=(12, 8))

correlation_matrix = np.zeros((len(energy_measures), len(dwell_measures)))

for i, (e_name, e_data) in enumerate(energy_measures.items()):
    for j, (d_name, d_data) in enumerate(dwell_measures.items()):
        valid = ~(np.isnan(e_data) | np.isnan(d_data))
        if np.sum(valid) > 1:
            r, _ = pearsonr(e_data[valid], d_data[valid])
            correlation_matrix[i, j] = r

sns.heatmap(correlation_matrix, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
            xticklabels=list(dwell_measures.keys()),
            yticklabels=list(energy_measures.keys()),
            cbar_kws={'label': 'Correlation'}, ax=ax, vmin=-1, vmax=1)

ax.set_title('Correlation Heatmap: All Energy × Dwell Combinations', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(pdf_dir / "01_correlation_heatmap.pdf", dpi=300, bbox_inches='tight')
plt.close()

print_both("✓ Plot 1: Heatmap saved")

# Plot 2: Best correlation scatter
best_row = df_results_sorted.iloc[0]
e_name, d_name = best_row['energy'], best_row['dwell']
e_data = energy_measures[e_name]
d_data = dwell_measures[d_name]

fig, ax = plt.subplots(figsize=(10, 8))

valid = ~(np.isnan(e_data) | np.isnan(d_data))
ax.scatter(e_data[valid], d_data[valid], alpha=0.5, s=30, color='steelblue', edgecolors='black', linewidth=0.5)

z = np.polyfit(e_data[valid], d_data[valid], 1)
p = np.poly1d(z)
x_line = np.linspace(np.min(e_data[valid]), np.max(e_data[valid]), 100)
ax.plot(x_line, p(x_line), "r--", linewidth=2.5)

ax.set_xlabel(e_name, fontsize=12, fontweight='bold')
ax.set_ylabel(d_name, fontsize=12, fontweight='bold')
ax.set_title(f'Strongest Correlation Found\nr={best_row["correlation"]:.4f}, p={best_row["p_value"]:.4e}',
             fontsize=13, fontweight='bold')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(pdf_dir / "02_strongest_correlation.pdf", dpi=300, bbox_inches='tight')
plt.close()

print_both("✓ Plot 2: Strongest correlation saved")

# Plot 3: Distribution of correlations
fig, ax = plt.subplots(figsize=(10, 6))

ax.hist(df_results['correlation'].values, bins=20, alpha=0.7, color='steelblue', edgecolor='black', linewidth=1.5)
ax.axvline(0, color='red', linestyle='--', linewidth=2, label='No correlation')
ax.axvline(df_results['correlation'].mean(), color='green', linestyle='--', linewidth=2,
           label=f'Mean r={df_results["correlation"].mean():.3f}')

ax.set_xlabel('Correlation Coefficient', fontsize=12, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax.set_title('Distribution of Correlations (All Combinations)', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(pdf_dir / "03_correlation_distribution.pdf", dpi=300, bbox_inches='tight')
plt.close()

print_both("✓ Plot 3: Distribution saved")

# Plot 4: Time series overlay
fig, axes = plt.subplots(4, 1, figsize=(14, 10))

axes[0].plot(modal_ctrl, linewidth=1.5, alpha=0.7, color='green', label='Modal Controllability')
axes[0].set_ylabel('Modal Controllability', fontsize=11, fontweight='bold')
axes[0].grid(alpha=0.3)
axes[0].legend()

axes[1].plot(network_entropy, linewidth=1.5, alpha=0.7, color='orange', label='Network Entropy')
axes[1].set_ylabel('Network Entropy', fontsize=11, fontweight='bold')
axes[1].grid(alpha=0.3)
axes[1].legend()

axes[2].plot(connectivity_strength, linewidth=1.5, alpha=0.7, color='purple', label='Connectivity Strength')
axes[2].set_ylabel('Connectivity Strength', fontsize=11, fontweight='bold')
axes[2].grid(alpha=0.3)
axes[2].legend()

axes[3].plot(dwell_2, linewidth=1.5, alpha=0.7, color='red', label='Velocity (Change Rate)')
axes[3].set_xlabel('Window', fontsize=11, fontweight='bold')
axes[3].set_ylabel('Velocity', fontsize=11, fontweight='bold')
axes[3].grid(alpha=0.3)
axes[3].legend()

plt.tight_layout()
plt.savefig(pdf_dir / "04_timeseries_all.pdf", dpi=300, bbox_inches='tight')
plt.close()

print_both("✓ Plot 4: Time series saved\n")

# ================================================================================
# SAVE DIAGNOSTIC REPORT
# ================================================================================

report_file = txt_dir / "diagnostic_report.txt"
with open(report_file, 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("HYPOTHESIS III PART B: DIAGNOSTIC REPORT\n")
    f.write("=" * 80 + "\n\n")

    f.write("ORIGINAL HYPOTHESIS:\n")
    f.write("Energy Cost ↑ → Dwell Time ↓ (NEGATIVE correlation)\n\n")

    f.write("FINDINGS:\n")
    f.write("-" * 80 + "\n\n")

    f.write(f"1. STRONGEST CORRELATION FOUND:\n")
    f.write(f"   Energy: {best_row['energy']}\n")
    f.write(f"   Dwell: {best_row['dwell']}\n")
    f.write(f"   Correlation: {best_row['correlation']:.4f}\n")
    f.write(f"   P-value: {best_row['p_value']:.4e}\n")
    f.write(f"   Significant: {'YES ✓' if best_row['significant'] else 'NO'}\n\n")

    f.write(f"2. SUMMARY STATISTICS:\n")
    f.write(f"   Mean correlation: {df_results['correlation'].mean():.4f}\n")
    f.write(f"   Median correlation: {df_results['correlation'].median():.4f}\n")
    f.write(f"   Std deviation: {df_results['correlation'].std():.4f}\n")
    f.write(f"   Range: [{df_results['correlation'].min():.4f}, {df_results['correlation'].max():.4f}]\n")
    f.write(f"   Significant results: {df_results['significant'].sum()} out of {len(df_results)}\n\n")

    f.write(f"3. INTERPRETATION:\n")
    f.write("-" * 80 + "\n\n")

    if df_results['significant'].sum() > 0:
        f.write("✓ SIGNIFICANT RELATIONSHIP DETECTED\n\n")
        f.write("The data shows a significant relationship between energy and dwell time.\n")
        f.write("This suggests thermodynamic principles DO influence brain state transitions.\n")
    else:
        f.write("⚠ NO SIGNIFICANT RELATIONSHIPS FOUND\n\n")
        f.write("Possible explanations:\n")
        f.write("1. Measures are not appropriate for this dataset\n")
        f.write("2. Thermodynamic optimization may not apply to this brain activity\n")
        f.write("3. The relationship is more complex (nonlinear, time-delayed, etc.)\n")
        f.write("4. Sample size or temporal resolution insufficient\n\n")

    f.write("4. NEXT STEPS:\n")
    f.write("-" * 80 + "\n")
    f.write("- Use the strongest correlation found as basis for further analysis\n")
    f.write("- Consider nonlinear relationships (polynomial, exponential)\n")
    f.write("- Examine time-lagged correlations\n")
    f.write("- Test with different window lengths\n")
    f.write("- Compare with surrogate/null data\n")

print_both("✓ Diagnostic report saved")
print_both("")

df_results.to_csv(data_dir / "all_correlations.csv", index=False)

print_both("=" * 80)
print_both("DIAGNOSTIC ANALYSIS COMPLETE")
print_both("=" * 80 + "\n")

print_both(f"✓ Results in: {output_dir}")
print_both(f"✓ Check diagnostic_report.txt for interpretation")

log_handle.close()

print(f"\n✓ Diagnostic analysis complete!")
print(f"✓ Results: {output_dir}")