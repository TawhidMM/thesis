import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt




morphology_csv_path = '/home/tawhid-mubashwir/Storage/morphlink/morphology_preprocessing/morphology_with_spot.csv'
morphology_df = pd.read_csv(morphology_csv_path)

MORPHOLOGY_COLS = [
    col for col in morphology_df.columns
    if col.startswith(('Nucleus:', 'Cell:', 'Cytoplasm:')) and
       morphology_df[col].dtype in [float, int]
]

# NON_MEAN_OD_COLS = [
#     col for col in morphology_df.columns
#     if 'OD' in col and not col.endswith('OD mean')
# ]
#
# MORPHOLOGY_COLS = list(set(MORPHOLOGY_COLS) - set(NON_MEAN_OD_COLS))


CELL_ID_COLUMN = 'Object ID'
NEW_CELL_ID_COLUMN = 'Cell ID'

cell_ids = morphology_df[CELL_ID_COLUMN]
morphology_df = morphology_df.rename(columns={CELL_ID_COLUMN: NEW_CELL_ID_COLUMN})

scaler = StandardScaler()
standardized_data = scaler.fit_transform(morphology_df[MORPHOLOGY_COLS])

standardized_df = pd.DataFrame(standardized_data, columns=MORPHOLOGY_COLS)

n_components = 15
pca = PCA(n_components=n_components)
pca_result = pca.fit_transform(standardized_df)

pca_df = pd.DataFrame(pca_result, columns=[f'PC{i+1}' for i in range(pca_result.shape[1])])
pca_df.insert(0, NEW_CELL_ID_COLUMN, cell_ids)

pca_df.to_csv(f'../input/morphology_pca_{pca_result.shape[1]}.csv', index=False)



# === Scree Plot (Variance Explained) ===
plt.figure(figsize=(10, 6))
plt.plot(np.cumsum(pca.explained_variance_ratio_)*100, marker='o')
plt.title('Scree Plot - Cumulative Explained Variance')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance (%)')
plt.grid(True)
plt.axhline(y=90, color='red', linestyle='--', label='90% Threshold')
plt.legend()
plt.tight_layout()
plt.savefig(f"pca_scree_plot_{n_components}.png", dpi=300)  # Save as image
plt.show()
