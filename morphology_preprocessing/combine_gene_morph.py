import pandas as pd

# Load PCA results
gene_pc_file = "/mnt/Drive E/Class Notes/L-4 T-1/Thesis/ScribbleDom/preprocessed_data/cancers/bcdc_ffpe/Principal_Components/CSV/pcs_15_from_bayesSpace_top_2000_HVGs.csv"
gene_df = pd.read_csv(gene_pc_file)  # from R
gene_df = gene_df.rename(columns={"Unnamed: 0": "assigned_spot"})
morph_df = pd.read_csv("mean_morphology_pca_result.csv")  # from previous Python PCA

# Merge on assigned_spot
merged_df = pd.merge(gene_df, morph_df, on="assigned_spot", suffixes=('_gene', '_morph'))

# Save combined PCA result
merged_df.to_csv("combined_gene_morphology_pca.csv", index=False)

print(f"gene PC shape: {gene_df.shape}")
print(f"morphology PC shape: {morph_df.shape}")
print(f"Combined shape: {merged_df.shape}")
print(f"Saved to combined_gene_morphology_pca.csv")


"""
gene PC shape: (2518, 16)
morphology PC shape: (2788, 41)
Combined shape: (2328, 56)

"""
