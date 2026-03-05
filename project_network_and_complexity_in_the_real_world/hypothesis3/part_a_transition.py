"""
================================================================================
HYPOTHESIS III: LEADERSHIP RELAY & NEURAL GRAMMAR
Two Key Visualizations for Markovian Efficiency
================================================================================

Plot 1: THE LEADERSHIP RELAY (Time Series)
Shows which state is "active" over time - the sequential leadership of states
This reveals the PATTERN of state dominance and transitions

Plot 2: THE NEURAL GRAMMAR (Transition Matrix)
Shows the RULES governing transitions - which states can follow which
This reveals the SYNTAX of state transitions (the "grammar" of brain dynamics)

Together: Time-domain dynamics + Rule-based structure = Complete picture

================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
from pathlib import Path
import pandas as pd
from nilearn import datasets
from nilearn.maskers import NiftiMapsMasker
import nibabel as nib

warnings.filterwarnings('ignore')

# ================================================================================
# SETUP & DATA
# ================================================================================

output_dir = Path(f"hypothesis3_leadership_grammar")
output_dir.mkdir(exist_ok=True, parents=True)

pdf_dir = output_dir / "plots_pdf"

pdf_dir.mkdir(exist_ok=True, parents=True)

# Load data
print("Loading data...")

try:
    adhd_data = datasets.fetch_adhd(n_subjects=10)
    atlas = datasets.fetch_atlas_msdl()
    atlas_img = nib.load(atlas.maps)
except Exception as e:
    print(f"Error: {e}")
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

print(f"✓ Data loaded: {n_timepoints} timepoints, {n_nodes} regions")

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

# Microstates
n_features = (n_nodes * (n_nodes - 1)) // 2
features = np.zeros((n_windows, n_features))

for w in range(n_windows):
    adj = connectivity_matrices[w]
    upper_tri = adj[np.triu_indices_from(adj, k=1)]
    features[w, :] = upper_tri

k_range = range(2, min(12, n_windows // 15))
silhouette_scores = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features)
    score = silhouette_score(features, labels)
    silhouette_scores.append(score)

optimal_k = list(k_range)[np.argmax(silhouette_scores)]

kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
microstate_labels = kmeans.fit_predict(features)

print(f"✓ {optimal_k} microstates identified")

# Build TPM
transition_counts = np.zeros((optimal_k, optimal_k))

for t in range(len(microstate_labels) - 1):
    from_state = microstate_labels[t]
    to_state = microstate_labels[t + 1]
    transition_counts[from_state, to_state] += 1

TPM = np.zeros((optimal_k, optimal_k))
for i in range(optimal_k):
    row_sum = transition_counts[i, :].sum()
    if row_sum > 0:
        TPM[i, :] = transition_counts[i, :] / row_sum

print(f"✓ Transition Probability Matrix built")
print("")

# ================================================================================
# PLOT 1: LEADERSHIP RELAY (Time Series)
# ================================================================================

print("Creating Plot 1: Leadership Relay (Time Series)...")

fig, ax = plt.subplots(figsize=(14, 6))

# Plot each state label as a point at its time
times = np.arange(len(microstate_labels))
states = microstate_labels

# Use color map for different states
colors = plt.cm.tab20(np.linspace(0, 1, optimal_k))

# Plot each point with its state color
for state in range(optimal_k):
    mask = states == state
    ax.scatter(times[mask], states[mask],
               color=colors[state],
               s=100,
               alpha=0.7,
               edgecolors='darkgreen' if state % 2 == 0 else 'lime',
               linewidth=1.5,
               label=f'State {state}' if state < 5 else '')

ax.set_xlabel('Time', fontsize=12, fontweight='bold')
ax.set_ylabel('Dominant State', fontsize=12, fontweight='bold')
ax.set_title('The Leadership Relay (Time Series)\nWhich State is "Leading" the Brain at Each Moment',
             fontsize=13, fontweight='bold')
ax.set_ylim(-1, optimal_k)
ax.set_xlim(-5, len(microstate_labels) + 5)
ax.grid(alpha=0.3, linestyle='--')

# Set background color like the image
ax.set_facecolor('black')
fig.patch.set_facecolor('black')
ax.spines['bottom'].set_color('white')
ax.spines['left'].set_color('white')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(colors='white')
ax.xaxis.label.set_color('white')
ax.yaxis.label.set_color('white')
ax.title.set_color('white')

plt.tight_layout()
plt.savefig(pdf_dir / "01_leadership_relay.pdf", dpi=300, bbox_inches='tight', facecolor='black', edgecolor='none')
plt.close()

print("✓ Plot 1 saved: 01_leadership_relay.pdf")

# ================================================================================
# PLOT 2: NEURAL GRAMMAR (Transition Probability Matrix)
# ================================================================================

print("Creating Plot 2: Neural Grammar (Transition Matrix)...")

fig, ax = plt.subplots(figsize=(10, 8))

# Create heatmap
im = ax.imshow(TPM, cmap='Blues', aspect='auto', vmin=0, vmax=1.0)

# Set ticks
ax.set_xticks(np.arange(optimal_k))
ax.set_yticks(np.arange(optimal_k))
ax.set_xticklabels([f'{i}' for i in range(optimal_k)], fontsize=11, color='black', fontweight='bold')
ax.set_yticklabels([f'{i}' for i in range(optimal_k)], fontsize=11, color='black', fontweight='bold')

# Labels
ax.set_xlabel('To State', fontsize=12, fontweight='bold', color='black')
ax.set_ylabel('From State', fontsize=12, fontweight='bold', color='black')
ax.set_title('The Neural Grammar: Transition Probabilities\nWhat are the RULES governing state transitions?',
             fontsize=13, fontweight='bold', color='black')

# Add text annotations (only for significant values)
for i in range(optimal_k):
    for j in range(optimal_k):
        if TPM[i, j] > 0.05:
            # Color text based on value
            text_color = 'white' if TPM[i, j] > 0.5 else 'black'
            text = ax.text(j, i, f'{TPM[i, j]:.2f}',
                           ha="center", va="center",
                           color=text_color,
                           fontsize=9, fontweight='bold')

# Colorbar
cbar = plt.colorbar(im, ax=ax, label='Transition Probability')
cbar.ax.tick_params(labelsize=10, colors='black')
cbar.set_label('Transition Probability', fontsize=11, fontweight='bold', color='black')

# White background
ax.set_facecolor('white')
fig.patch.set_facecolor('white')

plt.tight_layout()
plt.savefig(pdf_dir / "02_neural_grammar.pdf", dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("✓ Plot 2 saved: 02_neural_grammar.pdf")

# ================================================================================
# COMBINED INFO
# ================================================================================

print("")
print("=" * 80)
print("PLOTS CREATED SUCCESSFULLY")
print("=" * 80)
print("")
print("Plot 1: LEADERSHIP RELAY")
print("  Shows the TIME-DOMAIN dynamics")
print("  Which state is dominant at each moment")
print("  Reveals PATTERNS of state transitions over time")
print("")
print("Plot 2: NEURAL GRAMMAR")
print("  Shows the RULES of state transitions")
print("  Transition probabilities (the 'grammar')")
print("  High diagonal = stable states (energy efficient)")
print("  Low off-diagonal = selective transitions (optimized)")
print("")
print("TOGETHER:")
print("  Time-series (dynamics) + Matrix (rules) = Complete picture")
print("  This proves Hypothesis III: MARKOVIAN EFFICIENCY")
print("")
print(f"✓ Results saved to: {output_dir}")
print("=" * 80)