import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import adjusted_rand_score
from argument_parser import *


def plot_scatter(data_frame, output_img_path):
    plot_color = ["#F56867", "#FEB915", "#C798EE", "#59BE86", "#7495D3", "#D1D1D1", "#6D1A9C", "#15821E", "#3A84E6",
                  "#997273"]
    colors_to_plt = [plot_color[label % len(plot_color)] for label in data_frame.iloc[:,-3].values]

    plt.figure(figsize=(5, 5))
    plt.axis('off')

    # Coordinates: last two cols of cell_coord_df are assumed to be x and y
    x = data_frame[data_frame.columns[-2]].values
    y = (data_frame[data_frame.columns[-1]].max() - data_frame[data_frame.columns[-1]]).values

    # y = ( data_frame[data_frame.columns[-1]]).values

    plt.scatter(x, y, c=colors_to_plt, s=0.2, marker='o')
    plt.savefig(output_img_path, dpi=1200, bbox_inches='tight', pad_inches=0)



output_img_folder = f'{final_output_folder}/{dataset}/{samples[0]}/morphology/{scheme}'
sell_prediction_file = f'{final_output_folder}/{dataset}/{samples[0]}/morphology/{scheme}/final_cell_labels.csv'
cell_coords_file = f'preprocessed/{dataset}/{samples[0]}/cell_coords.csv'
cell_annotation_file = f'preprocessed/{dataset}/{samples[0]}/cell_level_annotation.csv'

# Read files
cell_prediction_df = pd.read_csv(sell_prediction_file, index_col=0)
cell_coord_df = pd.read_csv(cell_coords_file, index_col=0)
cell_annotation_df = pd.read_csv(cell_annotation_file, index_col=0)

# Align on index
# cell_prediction_df.sort_index(inplace=True)
# cell_coord_df.sort_index(inplace=True)
# cell_annotation_df.sort_index(inplace=True)

# Calculate ARI
def calc_ari(df_1, df_2):
    df_merged = pd.merge(df_1, df_2, left_index=True, right_index=True).dropna()

    cols = df_merged.columns
    for col in cols:
        df_merged[col] = df_merged[col].values.astype('int')


    return adjusted_rand_score(df_merged[cols[0]].values, df_merged[cols[1]].values)


ari = calc_ari(cell_annotation_df, cell_prediction_df)

final_output_ari = f'{final_output_folder}/{dataset}/{samples[0]}/morphology/{scheme}/ari.csv'

print(f"Ari for dataset:{dataset} scheme:{scheme} is: ", ari)
pd.DataFrame([{"ARI": ari}]).to_csv(final_output_ari)


# cell_prediction_df.iloc[:, -1] = cell_prediction_df.iloc[:, -1].replace({0: 1, 1: 2})

mask = torch.load(f"graph_representation/{dataset}/{samples[0]}/scribble_mask.pt", weights_only=True).numpy()

cell_scribble_df = pd.read_csv(f'preprocessed/{dataset}/{samples[0]}/cell_level_scribble.csv', index_col=0)
print("direct scribble: \n", cell_scribble_df.iloc[:, -1].value_counts())
cell_scribble_df = cell_scribble_df.loc[mask]
df_merged = pd.merge(cell_scribble_df, cell_coord_df, left_index=True, right_index=True).dropna()
plot_scatter(df_merged, f'{output_img_folder}/scribble.png')


scribble_annotations_df = cell_annotation_df.loc[mask]
print("annotation scribble: \n", scribble_annotations_df.iloc[:, -1].value_counts())
df_merged = pd.merge(scribble_annotations_df, cell_coord_df, left_index=True, right_index=True).dropna()
plot_scatter(df_merged, f'{output_img_folder}/scribble_annotation.png')


scribble_prediction_df = cell_prediction_df.loc[mask]
print("prediction scribble: \n", scribble_prediction_df.iloc[:, -1].value_counts())
df_merged = pd.merge(scribble_prediction_df, cell_coord_df, left_index=True, right_index=True).dropna()
plot_scatter(df_merged, f'{output_img_folder}/scribble_prediction.png')


# plotting predictions
df_merged = pd.merge(cell_prediction_df, cell_coord_df, left_index=True, right_index=True).dropna()
print("prediction + coord: ", len(df_merged))
plot_scatter(df_merged, f'{output_img_folder}/prediction.png')

# plotting manual annotations
df_merged = pd.merge(cell_annotation_df, cell_coord_df, left_index=True, right_index=True).dropna()
print("annotation + coord: ", len(df_merged))
plot_scatter(df_merged, f'{output_img_folder}/annotation.png')
