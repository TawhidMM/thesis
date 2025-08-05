from sklearn.preprocessing import StandardScaler
import pandas as pd

# Features to standardize (you can select only morphology features)

cells_file = 'mean_aggregated_spot_morphology.csv'
cells_df = pd.read_csv(cells_file)

morphology_cols = [
    col for col in cells_df.columns
    if col.startswith(('Nucleus:', 'Cell:', 'Cytoplasm:')) and cells_df[col].dtype in [float, int]
]

# Extract and standardize
scaler = StandardScaler()
standardized_data = scaler.fit_transform(cells_df[morphology_cols])

# Put back into a DataFrame
standardized_df = pd.DataFrame(standardized_data, columns=morphology_cols)

# Optionally, merge with other non-standardized columns
final_df = pd.concat([cells_df.drop(columns=morphology_cols), standardized_df], axis=1)

# Save the standardized DataFrame
method = cells_file.split('_')[0]
final_df.to_csv(f'{method}_standardized_spot_morphology.csv', index=False)
