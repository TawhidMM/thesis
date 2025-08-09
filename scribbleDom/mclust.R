
# Load library
library(mclust)

# --- Config ---
input_file <- "../input/morphology_pca_15.csv"         # CSV with Cell ID + PC1...PC15
output_file <- "../input/cell_mclust_backbone.csv"    # Output: Cell ID + Cluster Label
n_clusters <- 2                       # <-- Set your desired number of clusters
model_type <- "EEE"                   # Mclust model type

# --- Read and preprocess data ---
data <- read.csv(input_file)

# Separate PCs and Cell ID
cell_ids <- data$Cell.ID  # Column name will be adjusted to valid R variable
pcs <- data[, -1]         # Remove Cell ID, keep only PC columns

# Run Mclust with fixed number of clusters and model type
mclust_run <- Mclust(pcs, G = n_clusters, modelNames = model_type)

# Get cluster labels
cluster_labels <- mclust_run$classification

# Combine Cell ID with Cluster labels
output <- data.frame(Cell_ID = cell_ids, Cluster_Label = cluster_labels)

# Write result to CSV
write.csv(output, output_file, row.names = FALSE)

cat("✅ Mclust completed and saved to:", output_file, "\n")
