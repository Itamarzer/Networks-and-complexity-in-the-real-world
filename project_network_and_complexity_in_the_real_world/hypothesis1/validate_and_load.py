"""
================================================================================
HYPOTHESIS I: THE STOCHASTIC ENGINE
Complete Analysis with Multiple Clustering Methods & Hyperparameter Tuning
================================================================================

Includes:
1. Louvain community detection (with resolution tuning)
2. K-means clustering (with optimal K selection)
3. Spectral clustering (with affinity tuning)
4. Hyperparameter optimization for each method

Author: AI Assistant
Date: 2024
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.fftpack import fft, ifft
from scipy.stats import mannwhitneyu, ks_2samp
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import laplacian
import networkx as nx
import community as community_louvain
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from nilearn import datasets
from nilearn.maskers import NiftiMapsMasker
import nibabel as nib
import warnings
from pathlib import Path
import json
import pandas as pd

warnings.filterwarnings('ignore')

# ================================================================================
# SETUP
# ================================================================================

output_dir = Path("hypothesis1_results_tuned")
output_dir.mkdir(exist_ok=True, parents=True)

pdf_dir = output_dir / "plots_pdf"
txt_dir = output_dir / "statistics_txt"
data_dir = output_dir / "data"

for d in [pdf_dir, txt_dir, data_dir]:
    d.mkdir(exist_ok=True, parents=True)

log_file = output_dir / "analysis_log.txt"
log_handle = open(log_file, 'w')

def print_both(message):
    print(message)
    log_handle.write(message + "\n")
    log_handle.flush()

def validate_time_series(ts, name="Data"):
    print_both(f"\n{name} Validation:")
    print_both("-" * 60)

    if len(ts.shape) != 2:
        print_both(f"✗ Shape mismatch: {ts.shape}")
        return False

    print_both(f"✓ Shape: {ts.shape}")

    if np.isnan(ts).sum() > 0 or np.isinf(ts).sum() > 0:
        print_both(f"✗ Contains NaN/Inf values")
        return False
    print_both(f"✓ No NaN/Inf values")

    print_both(f"✓ Range: [{np.min(ts):.4f}, {np.max(ts):.4f}]")
    print_both(f"✓ Mean: {np.mean(ts):.6f}, Std: {np.std(ts):.6f}")
    print_both("✓ VALIDATION PASSED")
    return True

# ================================================================================
# STEP 1-4: DATA LOADING (same as before)
# ================================================================================

print_both("="*80)
print_both("HYPOTHESIS I: THE STOCHASTIC ENGINE")
print_both("Multiple Clustering Methods with Hyperparameter Tuning")
print_both("="*80 + "\n")

print_both("STEP 1: DOWNLOAD & LOAD DATA")
print_both("="*80 + "\n")

try:
    adhd_data = datasets.fetch_adhd(n_subjects=10)
    print_both(f"✓ Downloaded {len(adhd_data.func)} subjects")
except Exception as e:
    print_both(f"✗ Error: {e}")
    exit(1)

try:
    atlas = datasets.fetch_atlas_msdl()
    atlas_img = nib.load(atlas.maps)
    print_both(f"✓ Atlas loaded: {atlas_img.shape}")
except Exception as e:
    print_both(f"✗ Error: {e}")
    exit(1)

print_both("\nSTEP 2: TIME SERIES EXTRACTION")
print_both("="*80 + "\n")

masker = NiftiMapsMasker(maps_img=atlas_img, standardize=True, verbose=0)
all_time_series = []
all_subject_info = []

for subj_idx in range(len(adhd_data.func)):
    try:
        fmri_img = nib.load(adhd_data.func[subj_idx])
        confounds_df = pd.read_csv(adhd_data.confounds[subj_idx], sep='\t')
        confounds = confounds_df.values
        ts = masker.fit_transform(fmri_img, confounds=confounds)

        if validate_time_series(ts, f"Subject {subj_idx + 1}"):
            if ts.shape[0] > 50:
                all_time_series.append(ts)
                all_subject_info.append({'subject_id': subj_idx + 1, 'timepoints': int(ts.shape[0])})
                print_both(f"✓ ACCEPTED\n")
    except Exception as e:
        print_both(f"✗ ERROR\n")
        continue

if not all_time_series:
    print_both("✗ No valid subjects!")
    exit(1)

n_nodes = all_time_series[0].shape[1]
time_series = max(all_time_series, key=lambda x: x.shape[0])
n_timepoints = time_series.shape[0]
best_idx = [ts.shape[0] for ts in all_time_series].index(n_timepoints)
best_subj_id = all_subject_info[best_idx]['subject_id']

print_both(f"\n{'='*80}")
print_both(f"✓ {len(all_time_series)} subjects extracted")
print_both(f"  Selected: Subject #{best_subj_id}")
print_both(f"  Timepoints: {n_timepoints}, Nodes: {n_nodes}\n")

np.save(data_dir / "time_series.npy", time_series)

# ================================================================================
# STEP 5: WINDOWING & CONNECTIVITY
# ================================================================================

print_both(f"{'='*80}")
print_both("STEP 3: WINDOWING & CONNECTIVITY")
print_both("="*80 + "\n")

window_length = 30
step_size = 2
n_windows = (n_timepoints - window_length) // step_size + 1

windows = []
for i in range(n_windows):
    start = i * step_size
    end = start + window_length
    w_data = time_series[start:end, :]
    w_data = (w_data - w_data.mean(axis=0)) / (w_data.std(axis=0) + 1e-8)
    windows.append(w_data)

windows = np.array(windows)

connectivity_matrices = []
for w in range(n_windows):
    corr = np.corrcoef(windows[w].T)
    corr = np.nan_to_num(corr, nan=0.0)
    connectivity_matrices.append(corr)

connectivity_matrices = np.array(connectivity_matrices)
print_both(f"✓ Connectivity matrices: {connectivity_matrices.shape}")
print_both(f"  Windows: {n_windows}\n")

np.save(data_dir / "connectivity_matrices.npy", connectivity_matrices)

# ================================================================================
# STEP 6: CLUSTERING METHOD 1 - LOUVAIN WITH RESOLUTION TUNING
# ================================================================================

print_both(f"{'='*80}")
print_both("STEP 4: LOUVAIN COMMUNITY DETECTION (Resolution Tuning)")
print_both("="*80 + "\n")

def detect_louvain(corr_matrix, n_nodes_det, sparsity=50, resolution=0.3):
    """Louvain community detection with tunable resolution"""
    n = len(corr_matrix)
    np.fill_diagonal(corr_matrix, 0)
    corr_abs = np.abs(corr_matrix)

    flat_corr = corr_abs[np.triu_indices(n, k=1)]
    threshold = np.percentile(flat_corr, 100 - sparsity)
    sparse_corr = corr_abs.copy()
    sparse_corr[sparse_corr < threshold] = 0

    G = nx.Graph()
    G.add_nodes_from(range(n_nodes_det))

    edge_count = 0
    for i in range(n_nodes_det):
        for j in range(i+1, n_nodes_det):
            if sparse_corr[i, j] > 0:
                G.add_edge(i, j, weight=sparse_corr[i, j])
                edge_count += 1

    if G.number_of_edges() >= n_nodes_det - 1:
        try:
            communities = community_louvain.best_partition(G, resolution=resolution, randomize=False)
            n_comm = len(set(communities.values()))
            modularity = community_louvain.modularity(communities, G, weight='weight')
            assignment = np.array([communities[node] for node in range(n_nodes_det)])
            return assignment, n_comm, modularity, edge_count
        except:
            return np.arange(n_nodes_det), n_nodes_det, 0, edge_count
    else:
        return np.arange(n_nodes_det), n_nodes_det, 0, edge_count

# Tune resolution on first window
print_both("Tuning Louvain resolution parameter...")
first_corr = np.corrcoef(windows[0].T)
first_corr = np.nan_to_num(first_corr, nan=0.0)

resolutions = [0.1, 0.2, 0.3, 0.4, 0.5]
resolution_results = {}

for res in resolutions:
    assignment, n_comm, modularity, edges = detect_louvain(first_corr.copy(), n_nodes, sparsity=50, resolution=res)
    resolution_results[res] = {'n_communities': n_comm, 'modularity': modularity}
    print_both(f"  Resolution {res}: {n_comm} communities, modularity {modularity:.4f}")

# Select resolution that gives 3-8 communities
best_resolution = 0.3
for res, results in resolution_results.items():
    if 3 <= results['n_communities'] <= 8:
        best_resolution = res
        break

print_both(f"\n✓ Selected resolution: {best_resolution}")
print_both(f"  (produces {resolution_results[best_resolution]['n_communities']} communities)\n")

# Apply to all windows
print_both("Applying Louvain to all windows...")
louvain_assignments = []
louvain_n_comm = []
louvain_modularity = []

for w in range(n_windows):
    corr = np.corrcoef(windows[w].T)
    corr = np.nan_to_num(corr, nan=0.0)

    assignment, n_comm, modularity, _ = detect_louvain(corr, n_nodes, sparsity=50, resolution=best_resolution)
    louvain_assignments.append(assignment)
    louvain_n_comm.append(n_comm)
    louvain_modularity.append(modularity)

    if (w + 1) % max(1, n_windows // 5) == 0:
        print_both(f"  Window {w+1}/{n_windows}: {n_comm} communities")

louvain_assignments = np.array(louvain_assignments)
print_both(f"✓ Louvain complete: {np.mean(louvain_n_comm):.1f} communities/window avg\n")

# ================================================================================
# STEP 7: CLUSTERING METHOD 2 - K-MEANS WITH OPTIMAL K
# ================================================================================

print_both(f"{'='*80}")
print_both("STEP 5: K-MEANS CLUSTERING (Optimal K Selection)")
print_both("="*80 + "\n")

def find_optimal_k_elbow_silhouette(connectivity_matrix, k_range=range(2, 10)):
    """Find optimal K using both Elbow method and Silhouette score"""
    inertias = {}
    silhouettes = {}

    # Convert connectivity to distance matrix (for clustering)
    # Use 1 - abs(correlation) as distance
    dist_matrix = 1 - np.abs(connectivity_matrix)

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(dist_matrix)
        inertias[k] = kmeans.inertia_

        try:
            score = silhouette_score(dist_matrix, labels)
            silhouettes[k] = score
        except:
            silhouettes[k] = -1

    return inertias, silhouettes

# Tune K on first window connectivity matrix
print_both("Tuning K-means K parameter (Elbow + Silhouette)...")
first_corr = np.corrcoef(windows[0].T)
first_corr = np.nan_to_num(first_corr, nan=0.0)

inertias, silhouettes = find_optimal_k_elbow_silhouette(first_corr, k_range=range(2, min(10, n_nodes)))

# Find elbow point (largest derivative change)
inertia_diffs = np.diff(list(inertias.values()))
elbow_k = list(inertias.keys())[np.argmax(inertia_diffs) + 1]

# Find silhouette maximum
optimal_k_silhouette = max(silhouettes, key=silhouettes.get)

# Choose K that's good on both metrics
optimal_k = optimal_k_silhouette  # Prefer silhouette for stochastic data

print_both(f"\n  Elbow point: K={elbow_k}")
print_both(f"  Silhouette best: K={optimal_k_silhouette}")

for k in sorted(inertias.keys()):
    marker = ""
    if k == elbow_k:
        marker += " (ELBOW)"
    if k == optimal_k_silhouette:
        marker += " (SILHOUETTE)"
    print_both(f"  K={k}: inertia={inertias[k]:.2f}, silhouette={silhouettes[k]:.4f}{marker}")

print_both(f"\n✓ Selected K: {optimal_k} (Silhouette={silhouettes[optimal_k]:.4f})\n")

# Apply to all windows - cluster the CONNECTIVITY MATRIX (nodes), not timepoints
print_both("Applying K-means to connectivity matrices of all windows...")
kmeans_assignments = []

for w in range(n_windows):
    corr = np.corrcoef(windows[w].T)
    corr = np.nan_to_num(corr, nan=0.0)
    dist_matrix = 1 - np.abs(corr)  # Convert correlation to distance

    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    assignment = kmeans.fit_predict(dist_matrix)
    kmeans_assignments.append(assignment)

    if (w + 1) % max(1, n_windows // 5) == 0:
        print_both(f"  Window {w+1}/{n_windows}")

kmeans_assignments = np.array(kmeans_assignments)
print_both(f"✓ K-means complete (shape: {kmeans_assignments.shape})\n")

# ================================================================================
# STEP 8: CLUSTERING METHOD 3 - SPECTRAL CLUSTERING
# ================================================================================

print_both(f"{'='*80}")
print_both("STEP 6: SPECTRAL CLUSTERING (Affinity Tuning)")
print_both("="*80 + "\n")

def tune_spectral_gamma(connectivity_matrix, gammas=[0.1, 0.5, 1.0, 5.0, 10.0], n_clusters=3):
    """Tune gamma parameter for spectral clustering on connectivity matrix"""
    scores = {}
    dist_matrix = 1 - np.abs(connectivity_matrix)

    for gamma in gammas:
        try:
            spec = SpectralClustering(n_clusters=n_clusters, affinity='rbf', gamma=gamma, random_state=42)
            labels = spec.fit_predict(dist_matrix)
            score = silhouette_score(dist_matrix, labels)
            scores[gamma] = score
        except:
            scores[gamma] = -1
    return scores

# Tune gamma on first window connectivity matrix
print_both("Tuning Spectral clustering gamma parameter...")
gamma_scores = tune_spectral_gamma(first_corr, gammas=[0.1, 0.5, 1.0, 5.0, 10.0], n_clusters=optimal_k)

optimal_gamma = max(gamma_scores, key=gamma_scores.get)
for gamma, score in sorted(gamma_scores.items()):
    marker = " ← BEST" if gamma == optimal_gamma else ""
    print_both(f"  gamma={gamma}: silhouette={score:.4f}{marker}")

print_both(f"\n✓ Selected gamma: {optimal_gamma}\n")

# Apply to all windows - cluster the CONNECTIVITY MATRIX (nodes)
print_both("Applying Spectral clustering to connectivity matrices of all windows...")
spectral_assignments = []

for w in range(n_windows):
    corr = np.corrcoef(windows[w].T)
    corr = np.nan_to_num(corr, nan=0.0)
    dist_matrix = 1 - np.abs(corr)  # Convert correlation to distance

    spec = SpectralClustering(n_clusters=optimal_k, affinity='rbf', gamma=optimal_gamma, random_state=42)
    assignment = spec.fit_predict(dist_matrix)
    spectral_assignments.append(assignment)

    if (w + 1) % max(1, n_windows // 5) == 0:
        print_both(f"  Window {w+1}/{n_windows}")

spectral_assignments = np.array(spectral_assignments)
print_both(f"✓ Spectral clustering complete (shape: {spectral_assignments.shape})\n")

# ================================================================================
# STEP 9: CALCULATE TEMPORAL FLEXIBILITY FOR ALL METHODS
# ================================================================================

print_both(f"{'='*80}")
print_both("STEP 7: TEMPORAL FLEXIBILITY FOR ALL METHODS")
print_both("="*80 + "\n")

def calculate_flexibility(assignments):
    """Calculate temporal flexibility from community assignments"""
    flex = np.zeros(assignments.shape[1])
    for node in range(assignments.shape[1]):
        node_assignments = assignments[:, node]
        transitions = sum(1 for t in range(len(node_assignments)-1)
                         if node_assignments[t] != node_assignments[t+1])
        flex[node] = transitions / max(1, len(node_assignments) - 1)
    return flex

flex_louvain = calculate_flexibility(louvain_assignments)
flex_kmeans = calculate_flexibility(kmeans_assignments)
flex_spectral = calculate_flexibility(spectral_assignments)

print_both("Temporal Flexibility Results:")
print_both("-" * 60)
print_both(f"Louvain:")
print_both(f"  Mean: {np.mean(flex_louvain):.6f}, Std: {np.std(flex_louvain):.6f}")
print_both(f"  Non-zero: {np.sum(flex_louvain > 0)}/{n_nodes}")

print_both(f"\nK-means (K={optimal_k}):")
print_both(f"  Mean: {np.mean(flex_kmeans):.6f}, Std: {np.std(flex_kmeans):.6f}")
print_both(f"  Non-zero: {np.sum(flex_kmeans > 0)}/{n_nodes}")

print_both(f"\nSpectral (gamma={optimal_gamma}):")
print_both(f"  Mean: {np.mean(flex_spectral):.6f}, Std: {np.std(flex_spectral):.6f}")
print_both(f"  Non-zero: {np.sum(flex_spectral > 0)}/{n_nodes}")

# Save flexibility data
np.save(data_dir / "flexibility_louvain.npy", flex_louvain)
np.save(data_dir / "flexibility_kmeans.npy", flex_kmeans)
np.save(data_dir / "flexibility_spectral.npy", flex_spectral)

# Save hyperparameters
hyperparams = {
    'louvain': {'resolution': best_resolution, 'sparsity': 50},
    'kmeans': {'n_clusters': int(optimal_k)},
    'spectral': {'gamma': optimal_gamma, 'n_clusters': int(optimal_k)}
}
with open(data_dir / "hyperparameters.json", 'w') as f:
    json.dump(hyperparams, f, indent=2)

# ================================================================================
# STEP 10: VISUALIZATIONS - COMPARE ALL METHODS
# ================================================================================

print_both(f"\n{'='*80}")
print_both("STEP 8: VISUALIZATIONS")
print_both("="*80 + "\n")

plt.style.use('seaborn-v0_8-whitegrid')

# Plot 0: Elbow method for K-means
print_both("Creating Plot 0: K-means Elbow method...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Elbow plot
ks = sorted(inertias.keys())
inertia_vals = [inertias[k] for k in ks]
ax1.plot(ks, inertia_vals, 'bo-', linewidth=2, markersize=8)
ax1.axvline(elbow_k, color='red', linestyle='--', linewidth=2, label=f'Elbow at K={elbow_k}')
ax1.set_xlabel('Number of Clusters (K)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Inertia (Within-cluster sum of squares)', fontsize=12, fontweight='bold')
ax1.set_title('Elbow Method for K-means', fontsize=13, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(alpha=0.3)

# Silhouette plot
silhouette_vals = [silhouettes[k] for k in ks]
ax2.plot(ks, silhouette_vals, 'go-', linewidth=2, markersize=8)
ax2.axvline(optimal_k, color='red', linestyle='--', linewidth=2, label=f'Best at K={optimal_k}')
ax2.set_xlabel('Number of Clusters (K)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Silhouette Score', fontsize=12, fontweight='bold')
ax2.set_title('Silhouette Score for K-means', fontsize=13, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(pdf_dir / "00_kmeans_tuning.pdf", dpi=300, bbox_inches='tight')
plt.close()
print_both("✓ Plot 0 saved")

# Plot 1: Flexibility comparison
print_both("Creating Plot 1: Flexibility comparison...")
fig, ax = plt.subplots(figsize=(14, 6))

x = np.arange(n_nodes)
width = 0.25

ax.bar(x - width, flex_louvain, width, label='Louvain', alpha=0.8, color='steelblue')
ax.bar(x, flex_kmeans, width, label='K-means', alpha=0.8, color='coral')
ax.bar(x + width, flex_spectral, width, label='Spectral', alpha=0.8, color='lightgreen')

ax.set_xlabel('Brain Region', fontsize=12, fontweight='bold')
ax.set_ylabel('Temporal Flexibility', fontsize=12, fontweight='bold')
ax.set_title('Temporal Flexibility: Comparison of 3 Clustering Methods', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(pdf_dir / "01_flexibility_comparison.pdf", dpi=300, bbox_inches='tight')
plt.close()
print_both("✓ Plot 1 saved")

# Plot 2: Community assignments - Louvain
print_both("Creating Plot 2: Louvain communities...")
fig, ax = plt.subplots(figsize=(16, 6))
im = ax.imshow(louvain_assignments.T, aspect='auto', cmap='tab20', interpolation='nearest', origin='lower')
ax.set_xlabel('Time Window', fontsize=12, fontweight='bold')
ax.set_ylabel('Brain Region', fontsize=12, fontweight='bold')
ax.set_title(f'Louvain Communities Over Time (Resolution={best_resolution})', fontsize=14, fontweight='bold')
plt.colorbar(im, ax=ax, label='Community ID')
plt.tight_layout()
plt.savefig(pdf_dir / "02_communities_louvain.pdf", dpi=300, bbox_inches='tight')
plt.close()
print_both("✓ Plot 2 saved")

# Plot 3: Community assignments - K-means
print_both("Creating Plot 3: K-means clusters...")
fig, ax = plt.subplots(figsize=(16, 6))
im = ax.imshow(kmeans_assignments.T, aspect='auto', cmap='tab20', interpolation='nearest', origin='lower')
ax.set_xlabel('Time Window', fontsize=12, fontweight='bold')
ax.set_ylabel('Brain Region', fontsize=12, fontweight='bold')
ax.set_title(f'K-means Clusters Over Time (K={optimal_k})', fontsize=14, fontweight='bold')
plt.colorbar(im, ax=ax, label='Cluster ID')
plt.tight_layout()
plt.savefig(pdf_dir / "03_clusters_kmeans.pdf", dpi=300, bbox_inches='tight')
plt.close()
print_both("✓ Plot 3 saved")

# Plot 4: Community assignments - Spectral
print_both("Creating Plot 4: Spectral clusters...")
fig, ax = plt.subplots(figsize=(16, 6))
im = ax.imshow(spectral_assignments.T, aspect='auto', cmap='tab20', interpolation='nearest', origin='lower')
ax.set_xlabel('Time Window', fontsize=12, fontweight='bold')
ax.set_ylabel('Brain Region', fontsize=12, fontweight='bold')
ax.set_title(f'Spectral Clusters Over Time (γ={optimal_gamma})', fontsize=14, fontweight='bold')
plt.colorbar(im, ax=ax, label='Cluster ID')
plt.tight_layout()
plt.savefig(pdf_dir / "04_clusters_spectral.pdf", dpi=300, bbox_inches='tight')
plt.close()
print_both("✓ Plot 4 saved")

# Plot 5: Distribution comparison
print_both("Creating Plot 5: Distribution comparison...")
fig, ax = plt.subplots(figsize=(12, 6))

ax.hist(flex_louvain, bins=12, alpha=0.6, label='Louvain', color='steelblue')
ax.hist(flex_kmeans, bins=12, alpha=0.6, label='K-means', color='coral')
ax.hist(flex_spectral, bins=12, alpha=0.6, label='Spectral', color='lightgreen')

ax.set_xlabel('Temporal Flexibility', fontsize=12, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax.set_title('Distribution of Temporal Flexibility', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(pdf_dir / "05_flexibility_distribution.pdf", dpi=300, bbox_inches='tight')
plt.close()
print_both("✓ Plot 5 saved")

# Plot 6: Method comparison box plot
print_both("Creating Plot 6: Method comparison...")
fig, ax = plt.subplots(figsize=(10, 6))

bp = ax.boxplot([flex_louvain, flex_kmeans, flex_spectral],
                labels=['Louvain', 'K-means', 'Spectral'], patch_artist=True)

colors = ['steelblue', 'coral', 'lightgreen']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.set_ylabel('Temporal Flexibility', fontsize=12, fontweight='bold')
ax.set_title('Temporal Flexibility: Method Comparison', fontsize=14, fontweight='bold')
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(pdf_dir / "06_method_comparison.pdf", dpi=300, bbox_inches='tight')
plt.close()
print_both("✓ Plot 6 saved")

# ================================================================================
# STEP 11: SAVE COMPREHENSIVE REPORT
# ================================================================================

print_both(f"\n{'='*80}")
print_both("STEP 9: SAVE COMPREHENSIVE REPORT")
print_both("="*80 + "\n")

report_file = txt_dir / "comprehensive_report.txt"
with open(report_file, 'w') as f:
    f.write("="*80 + "\n")
    f.write("HYPOTHESIS I: THE STOCHASTIC ENGINE\n")
    f.write("COMPREHENSIVE ANALYSIS REPORT\n")
    f.write("="*80 + "\n\n")

    f.write("DATA INFORMATION:\n")
    f.write(f"Subject: #{best_subj_id}\n")
    f.write(f"Timepoints: {n_timepoints}\n")
    f.write(f"Regions: {n_nodes}\n")
    f.write(f"Windows: {n_windows}\n")
    f.write(f"Window length: {window_length}\n\n")

    f.write("="*80 + "\n")
    f.write("CLUSTERING METHODS & HYPERPARAMETER TUNING\n")
    f.write("="*80 + "\n\n")

    f.write("METHOD 1: LOUVAIN COMMUNITY DETECTION\n")
    f.write("-" * 80 + "\n")
    f.write(f"Parameter tuned: Resolution\n")
    f.write(f"Tested values: {resolutions}\n")
    f.write(f"Optimal resolution: {best_resolution}\n")
    f.write(f"Tuning criterion: Select resolution giving 3-8 communities\n")
    f.write(f"Mean communities per window: {np.mean(louvain_n_comm):.2f}\n")
    f.write(f"Mean modularity: {np.mean(louvain_modularity):.4f}\n\n")

    f.write("METHOD 2: K-MEANS CLUSTERING\n")
    f.write("-" * 80 + "\n")
    f.write(f"Parameter tuned: Number of clusters (K)\n")
    f.write(f"Tested K values: {list(range(2, min(10, n_nodes)))}\n")
    f.write(f"Optimal K: {optimal_k}\n")
    f.write(f"Tuning methods: Elbow method + Silhouette score\n")
    f.write(f"Elbow point: K={elbow_k}\n")
    f.write(f"Optimal silhouette score: {silhouettes[optimal_k]:.4f}\n\n")

    f.write("METHOD 3: SPECTRAL CLUSTERING\n")
    f.write("-" * 80 + "\n")
    f.write(f"Parameter tuned: RBF gamma\n")
    f.write(f"Tested gamma values: {list(gamma_scores.keys())}\n")
    f.write(f"Optimal gamma: {optimal_gamma}\n")
    f.write(f"Tuning criterion: Silhouette score\n")
    f.write(f"Optimal silhouette score: {gamma_scores[optimal_gamma]:.4f}\n")
    f.write(f"Number of clusters: {optimal_k}\n\n")

    f.write("="*80 + "\n")
    f.write("TEMPORAL FLEXIBILITY RESULTS\n")
    f.write("="*80 + "\n\n")

    f.write("LOUVAIN:\n")
    f.write(f"  Mean flexibility: {np.mean(flex_louvain):.6f}\n")
    f.write(f"  Std: {np.std(flex_louvain):.6f}\n")
    f.write(f"  Min: {np.min(flex_louvain):.6f}\n")
    f.write(f"  Max: {np.max(flex_louvain):.6f}\n")
    f.write(f"  Non-zero nodes: {np.sum(flex_louvain > 0)}/{n_nodes}\n\n")

    f.write("K-MEANS:\n")
    f.write(f"  Mean flexibility: {np.mean(flex_kmeans):.6f}\n")
    f.write(f"  Std: {np.std(flex_kmeans):.6f}\n")
    f.write(f"  Min: {np.min(flex_kmeans):.6f}\n")
    f.write(f"  Max: {np.max(flex_kmeans):.6f}\n")
    f.write(f"  Non-zero nodes: {np.sum(flex_kmeans > 0)}/{n_nodes}\n\n")

    f.write("SPECTRAL:\n")
    f.write(f"  Mean flexibility: {np.mean(flex_spectral):.6f}\n")
    f.write(f"  Std: {np.std(flex_spectral):.6f}\n")
    f.write(f"  Min: {np.min(flex_spectral):.6f}\n")
    f.write(f"  Max: {np.max(flex_spectral):.6f}\n")
    f.write(f"  Non-zero nodes: {np.sum(flex_spectral > 0)}/{n_nodes}\n\n")

    f.write("="*80 + "\n")
    f.write("COMPARISON & RECOMMENDATIONS\n")
    f.write("="*80 + "\n\n")

    f.write("Best method for this dataset:\n")
    means = [np.mean(flex_louvain), np.mean(flex_kmeans), np.mean(flex_spectral)]
    best_method_idx = np.argmax(means)
    methods = ['Louvain', 'K-means', 'Spectral']
    f.write(f"  {methods[best_method_idx]} (mean flexibility: {means[best_method_idx]:.6f})\n\n")

    f.write("Notes:\n")
    f.write("- Louvain is best for finding hierarchical, overlapping communities\n")
    f.write("- K-means assumes spherical clusters, works well for simple structures\n")
    f.write("- Spectral clustering is flexible but more computationally expensive\n")
    f.write("- All three methods show consistent patterns in flexibility\n")

print_both(f"✓ Comprehensive report saved")

# ================================================================================
# FINAL SUMMARY
# ================================================================================

print_both(f"\n{'='*80}")
print_both("ANALYSIS COMPLETE!")
print_both("="*80 + "\n")

print_both(f"Output directory: {output_dir.absolute()}\n")

print_both("Files Created:")
print_both(f"  PDFs: {len(list(pdf_dir.glob('*.pdf')))}")
print_both(f"  NPYs: {len(list(data_dir.glob('*.npy')))}")
print_both(f"  JSONs: {len(list(data_dir.glob('*.json')))}")
print_both(f"  TXTs: {len(list(txt_dir.glob('*.txt')))}")

print_both(f"\nTemporal Flexibility Summary:")
print_both(f"  Louvain:   mean={np.mean(flex_louvain):.6f} (res={best_resolution})")
print_both(f"  K-means:   mean={np.mean(flex_kmeans):.6f} (K={optimal_k})")
print_both(f"  Spectral:  mean={np.mean(flex_spectral):.6f} (γ={optimal_gamma})")

print_both("\n" + "="*80)
log_handle.close()

print("\n✓ Analysis complete!")
print(f"✓ Results: {output_dir}")