import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import skew, probplot
from matplotlib.backends.backend_pdf import PdfPages


cells_file = 'median_aggregated_spot_morphology.csv'
cells_df = pd.read_csv(cells_file)

morphology_cols = [
    col for col in cells_df.columns
    if col.startswith(('Nucleus:', 'Cell:', 'Cytoplasm:')) and cells_df[col].dtype in [float, int]
]


# Pick a subset of features to inspect (or all)
features_to_check = morphology_cols

with PdfPages(f"{cells_file}.pdf") as pdf:
    for col in features_to_check:
        plt.figure(figsize=(10, 4))

        # Histogram with KDE
        plt.subplot(1, 2, 1)
        sns.histplot(cells_df[col].dropna(), kde=True, bins=40)
        plt.title(f"{col} - Histogram")

        # Q-Q Plot
        plt.subplot(1, 2, 2)
        probplot(cells_df[col].dropna(), dist="norm", plot=plt)
        plt.title(f"{col} - Q-Q Plot")

        # Add skewness
        plt.suptitle(f"{col} | Skewness = {skew(cells_df[col].dropna()):.2f}")

        # Save current page to PDF
        pdf.savefig()
        plt.close()