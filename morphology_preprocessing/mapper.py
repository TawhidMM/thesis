import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

# Load the Visium spot data (Space Ranger output)
spots_tissue_position_file = '/mnt/Drive E/Class Notes/L-4 T-1/Thesis/ScribbleDom/preprocessed_data/cancers/bcdc_ffpe/reading_h5/spatial/tissue_positions_list.csv'
spot_annotation_file = '/mnt/Drive E/Class Notes/L-4 T-1/Thesis/ScribbleDom/preprocessed_data/cancers/bcdc_ffpe/manual_annotations.csv'

tissue_position_df = pd.read_csv(spots_tissue_position_file, header=None, index_col=0)
spot_annotations_df = pd.read_csv(spot_annotation_file, header=None, index_col=0)

in_tissue_spots = tissue_position_df.index.intersection(spot_annotations_df.index)
tissue_position_df = tissue_position_df.loc[in_tissue_spots]

print(tissue_position_df.shape)
print(tissue_position_df.head())
# Extract pixel coordinates: column 4 = X, column 5 = Y
# These are already in pixel units
spot_coords = tissue_position_df.iloc[:, -2:].values


# Load QuPath cell centroid coordinates (CSV with columns 'x', 'y' in the same pixel space)
cells_file = '/mnt/Drive E/Class Notes/L-4 T-1/Thesis/ScribbleDom/preprocessed_data/cancers/bcdc_ffpe/Coordinates/morphology_px.csv'
X_COL_NAME = 'centroid_x_px'
Y_COL_NAME = 'centroid_y_px'
cells_df = pd.read_csv(cells_file)
cell_coords = cells_df[[X_COL_NAME, Y_COL_NAME]].values

# Determine the Visium spot radius (in pixels).
# Try to read from scalefactors_json.json; otherwise use a default (e.g. ~45 px).
try:
    # with open('scalefactors_json.json') as f:
    #     scalefactors = json.load(f)
    # spot_diameter = float(scalefactors.get('spot_diameter_fullres', np.nan))
    spot_diameter = 188.56998854645946
    if np.isnan(spot_diameter):
        raise KeyError("spot_diameter_fullres missing")
    spot_radius = spot_diameter / 2.0
except Exception as e:
    print("Warning: using default spot radius (fallback).", e)
    spot_radius = 45.0
print(f"Using spot radius = {spot_radius:.1f} pixels.")


# Build a KD-tree on the spot centers for efficient neighbor search
tree = cKDTree(spot_coords)
# Query the nearest and second-nearest spot for each cell
dists, idxs = tree.query(cell_coords, k=2)

# Take the index of the nearest spot for each cell
closest_idx = idxs[:, 0].copy()
# If the two nearest spots are exactly equidistant, choose the smaller index
tie_mask = np.isclose(dists[:, 0], dists[:, 1])
closest_idx[tie_mask] = np.minimum(idxs[tie_mask, 0], idxs[tie_mask, 1])

# Filter cells to those within the spot radius
in_radius = (dists[:, 0] <= spot_radius)
cells_in_radius = cells_df[in_radius].reset_index(drop=True)
assigned_spot_idx = closest_idx[in_radius]

cells_in_radius["assigned_spot"] = tissue_position_df.index[assigned_spot_idx].values
cells_in_radius.to_csv('morphology_with_spot.csv', index=False)


# # Save the mapping table
# mapped_df.to_csv('cell_to_spot_mapping.csv', index=False)
print(f"Total spots: {len(tissue_position_df)}")
print(f"Total cells: {len(cells_df)}")
print(f"Mapped {len(cells_in_radius)} cells to spots (output saved).")




# # Plot spot centers
# plt.scatter(spot_coords[:,0], spot_coords[:,1], facecolors='none', edgecolors='red', label='Spots')
# # Draw circles around each spot center
# for x, y in spot_coords:
#     circle = plt.Circle((x, y), spot_radius, color='red', fill=False, alpha=0.3)
#     plt.gca().add_patch(circle)
# # Plot cell centroids
# plt.scatter(cells_in_radius[X_COL_NAME], cells_in_radius[Y_COL_NAME], s=5, c='blue', label='Cells')
# plt.legend(); plt.gca().invert_yaxis()
# plt.savefig("overlay_plot.png")
# plt.show()
