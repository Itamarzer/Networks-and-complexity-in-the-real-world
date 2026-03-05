import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path

from nilearn import datasets, plotting, image
from nilearn.maskers import NiftiMapsMasker
from nilearn.connectome import ConnectivityMeasure

print("1. Downloading / Loading Brain Data...")
# Load ADHD Functional Data & MSDL Atlas
adhd_data = datasets.fetch_adhd(n_subjects=1)
msdl_atlas = datasets.fetch_atlas_msdl()
fmri_img = adhd_data.func[0]
confounds = adhd_data.confounds[0]

# Setup PDF to save the plots
pdf_filename = "brain_network_visualizations.pdf"
pdf_pages = PdfPages(pdf_filename)

# ==========================================
# PLOT 1: GLASS BRAIN
# ==========================================
print("2. Generating Glass Brain Plot...")
stat_img = image.index_img(msdl_atlas.maps, 0)

fig1 = plt.figure(figsize=(8, 4))

display_glass = plotting.plot_glass_brain(
    stat_img,
    title="Plot 1: Glass Brain (Default Mode Network)",
    black_bg=True,
    display_mode="xz",
    threshold='auto',
    figure=fig1
)
pdf_pages.savefig(fig1)
plt.close(fig1)

# ==========================================
# PLOT 2: CONNECTOME
# ==========================================
print("3. Generating Connectome Plot...")
masker = NiftiMapsMasker(maps_img=msdl_atlas.maps, standardize=True, verbose=0)
time_series = masker.fit_transform(fmri_img, confounds=confounds)

correlation_measure = ConnectivityMeasure(kind='correlation')
correlation_matrix = correlation_measure.fit_transform([time_series])[0]
np.fill_diagonal(correlation_matrix, 0)

coords = msdl_atlas.region_coords

fig2 = plt.figure(figsize=(8, 4))

display_connectome = plotting.plot_connectome(
    correlation_matrix,
    coords,
    title="Plot 2: Functional Connectome",
    edge_threshold="80%",
    node_size=50,
    figure=fig2
)
pdf_pages.savefig(fig2)
plt.close(fig2)

# ==========================================
# PLOT 3: DESTRIEUX ATLAS VISUALIZATION
# ==========================================
print("4. Generating Destrieux Atlas Plot...")
destrieux_atlas = datasets.fetch_atlas_destrieux_2009(lateralized=True)

# The Destrieux atlas is volumetric (NIfTI), so use plot_roi instead of surface plotting
atlas_maps = destrieux_atlas['maps']

fig3 = plt.figure(figsize=(8, 4))

display_atlas = plotting.plot_roi(
    atlas_maps,
    title="Plot 3: Destrieux Atlas (Volumetric Parcellation)",
    display_mode="xz",
    cmap="tab20",
    figure=fig3
)
pdf_pages.savefig(fig3)
plt.close(fig3)

# ==========================================
# PLOT 4: ROI / PARCELLATION MAP
# ==========================================
print("5. Generating ROI Parcellation Plot...")
ho_atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
ward_labels_img = ho_atlas.maps

output_dir = Path.cwd() / "results" / "plot_data_driven_parcellations"
output_dir.mkdir(exist_ok=True, parents=True)
ward_labels_img.to_filename(output_dir / "ward_parcellation_mock.nii.gz")

fig4 = plt.figure(figsize=(8, 4))

display_roi = plotting.plot_roi(
    ward_labels_img,
    title="Plot 4: Discrete Parcellation (ROI)",
    display_mode="xz",
    cmap="Paired",
    figure=fig4
)
pdf_pages.savefig(fig4)
plt.close(fig4)

# ==========================================
# FINALIZE
# ==========================================
pdf_pages.close()
print(f"\n✅ SUCCESS! All 4 plots saved to: {pdf_filename}")