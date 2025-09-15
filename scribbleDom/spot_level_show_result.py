import glob
import os

import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import adjusted_rand_score
from argument_parser import *


def plot_scatter(data_frame, output_img_path):
    for col in data_frame.columns:
        data_frame[col] = data_frame[col].values.astype('int')

    plot_color = ["#F56867", "#FEB915", "#C798EE", "#59BE86", "#7495D3", "#D1D1D1", "#6D1A9C", "#15821E", "#3A84E6",
                  "#997273"]
    colors_to_plt = [plot_color[label % len(plot_color)] for label in data_frame.iloc[:,-3].values]

    plt.figure(figsize=(5, 5))
    plt.axis('off')

    # Coordinates: last two cols of cell_coord_df are assumed to be x and y
    x = data_frame[data_frame.columns[-2]].values
    y = (data_frame[data_frame.columns[-1]].max() - data_frame[data_frame.columns[-1]]).values

    # y = ( data_frame[data_frame.columns[-1]]).values

    plt.scatter(x, y, c=colors_to_plt, s=5, marker='o', alpha=1)
    plt.savefig(output_img_path, dpi=1200, bbox_inches='tight', pad_inches=0)
    plt.close()


# Calculate ARI
def calc_ari(df_1, df_2):
    df_merged = pd.merge(df_1, df_2, left_index=True, right_index=True).dropna()

    cols = df_merged.columns
    for col in cols:
        df_merged[col] = df_merged[col].values.astype('int')


    return adjusted_rand_score(df_merged[cols[0]].values, df_merged[cols[1]].values)


def show_results(folder_path):
    # final_cell_labels.csv  final_meta_learner_predictions.csv
    spot_predictions_df = pd.read_csv(f"{folder_path}/final_cell_labels.csv", index_col=0)

    spots_tissue_position_file = f'/mnt/Drive E/Class Notes/L-4 T-1/Thesis/ScribbleDom/preprocessed_data/{dataset}/{samples[0]}/reading_h5/spatial/tissue_positions_list.csv'
    spot_annotation_file = f'/mnt/Drive E/Class Notes/L-4 T-1/Thesis/ScribbleDom/preprocessed_data/{dataset}/{samples[0]}/manual_annotations.csv'

    tissue_position_df = pd.read_csv(spots_tissue_position_file,index_col=0)
    spot_annotations_df = pd.read_csv(spot_annotation_file,index_col=0)

    spot_annotations_df.sort_index(inplace=True)
    spot_predictions_df.sort_index(inplace=True)
    tissue_position_df.sort_index(inplace=True)

    in_tissue_spots = tissue_position_df.index.intersection(spot_annotations_df.index)
    tissue_position_df = tissue_position_df.loc[in_tissue_spots]

    mask = torch.load(f"graph_representation/{dataset}/{samples[0]}/mask.pt", weights_only=True)
    mask = mask.numpy()

    spot_annotations_df = spot_annotations_df.loc[mask]
    tissue_position_df = tissue_position_df.loc[mask]

    spot_coords_df = tissue_position_df.iloc[:, [-1, -2]]
    spot_coords_df.sort_index(inplace=True)

    ari = calc_ari(spot_annotations_df, spot_predictions_df)

    print(f"Ari for dataset:{dataset} scheme:{scheme} is: ", ari)
    pd.DataFrame([{"ARI": ari}]).to_csv(f'{folder_path}/ari.csv', index=False)

    # plotting predictions
    df_merged = pd.merge(spot_predictions_df, spot_coords_df, left_index=True, right_index=True).dropna()
    plot_scatter(df_merged, f'{folder_path}/prediction.png')

    # plotting manual annotations
    df_merged = pd.merge(spot_annotations_df, spot_coords_df, left_index=True, right_index=True).dropna()
    plot_scatter(df_merged, f'{folder_path}/annotation.png')


for sample in samples:
    model_output_directory = f"{output_data_path}/{dataset}/{sample}/morphology-new/{scheme}"
    output_directories = sorted(glob.glob(os.path.join(model_output_directory, "*")))

    for output_dir in output_directories:
        print(output_dir)
        show_results(output_dir)

