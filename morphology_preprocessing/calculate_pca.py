import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
#
# # === Load aggregated and optionally standardized data ===
csv_file = "mean_standardized_spot_morphology.csv"
df = pd.read_csv(csv_file)
spot_ids = df['assigned_spot']
features = df.drop(columns=['assigned_spot'])

# === Run PCA ===
pca = PCA()
pca_result = pca.fit_transform(features)

# === Save PCA results with spot IDs ===
pca_df = pd.DataFrame(pca_result, columns=[f'PC{i+1}' for i in range(pca_result.shape[1])])
pca_df.insert(0, 'assigned_spot', spot_ids)

method = csv_file.split('_')[0]
pca_df.to_csv(f"{method}_morphology_pca_result.csv", index=False)

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
plt.savefig(f"{method}_pca_scree_plot.png", dpi=300)  # Save as image
plt.show()

