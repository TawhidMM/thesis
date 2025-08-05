import pandas as pd


cell_features_path = f'/home/tawhid-mubashwir/Storage/morphlink/morphology_preprocessing/morphology_with_spot.csv'  # Path to the cell features CSV file
scribble_path = f'/mnt/Drive E/Class Notes/L-4 T-1/Thesis/ScribbleDom/preprocessed_data/cancers/bcdc_ffpe/manual_scribble.csv'      # Path to the spot labels CSV file
manual_annotation_path =  "/mnt/Drive E/Class Notes/L-4 T-1/Thesis/ScribbleDom/preprocessed_data/cancers/bcdc_ffpe/manual_annotations.csv"

output_path = f'../input/cell_level_scribble.csv'  # Path to save the output

"""
need to handle the input folder in better way
"""

# Load the cell features and spot scribbles
cell_df = pd.read_csv(cell_features_path)
spot_scribble_df = pd.read_csv(scribble_path)
spot_manual_annotations_df = pd.read_csv(manual_annotation_path)

# Fix unnamed column if needed
spot_scribble_df.columns = ['spot_id', 'label']
spot_manual_annotations_df.columns = ['spot_id', 'label']

# Merge on assigned_spot → node gets label if its spot is labeled
cell_scribble_merged_df = cell_df.merge(spot_scribble_df, left_on='assigned_spot', right_on='spot_id', how='inner')
cell_annotation_merged_df = cell_df.merge(spot_manual_annotations_df, left_on='assigned_spot', right_on='spot_id', how='inner')


# Extract node_id and label (-1 for missing)
cell_scribble_df = pd.DataFrame({
    'Cell ID': cell_scribble_merged_df['Object ID'],
    'scribble_label': cell_scribble_merged_df['label'].fillna(-1).astype(int)
})

cell_annotation_df = pd.DataFrame({
    'Cell ID': cell_annotation_merged_df['Object ID'],
    'label': cell_annotation_merged_df['label'].fillna(-1).astype(int)
})

print("count values for cell scribbles: ", cell_scribble_df['scribble_label'].value_counts())
print("count values for cell annotation: ", cell_annotation_df['label'].value_counts())

print("size", len(cell_annotation_df))
print("size", len(cell_scribble_df))

# Save to CSV
cell_scribble_df.to_csv(output_path, index=False)
cell_annotation_df.to_csv(f'../input/cell_level_annotation.csv', index=False)
print(f"✅ Saved node-to-scribble map to {output_path}")

