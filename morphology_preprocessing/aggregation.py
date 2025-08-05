import pandas as pd


cells_file = 'morphology_with_spot.csv'
cells_df = pd.read_csv(cells_file)

# Assuming cells_df has a 'barcode' column and all morphology features
# Select only numeric morphology features for aggregation
morphology_cols = [
    col for col in cells_df.columns
    if col.startswith(('Nucleus:', 'Cell:', 'Cytoplasm:')) and cells_df[col].dtype in [float, int]
]

# List of aggregation methods you want to try
aggregation_methods = ['mean', 'median']

for method in aggregation_methods:
    spot_aggregated_df = (
        cells_df.groupby('assigned_spot')[morphology_cols]
        .agg(method)
        .reset_index()
    )

    spot_aggregated_df.to_csv(f'{method}_aggregated_spot_morphology.csv', index=False)

    print(f"Aggregation method: {method}")
    print(f"Mapped cells: {len(cells_df)}")
    print(f"Aggregated spots: {len(spot_aggregated_df)}\n")

# mapped cells: 67422
# aggregated spots: 2788