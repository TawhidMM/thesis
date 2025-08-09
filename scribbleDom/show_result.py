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

    plt.scatter(x, y, c=colors_to_plt, s=0.2, marker='o')
    plt.savefig(output_img_path, dpi=1200, bbox_inches='tight', pad_inches=0)



output_img_folder = f'{final_output_folder}/{dataset}/{samples[0]}/morphology/{scheme}'
sell_prediction_file = f'{final_output_folder}/{dataset}/{samples[0]}/morphology/{scheme}/final_cell_labels.csv'

# Read files
cell_prediction_df = pd.read_csv(sell_prediction_file, index_col=0)
cell_coord_df = pd.read_csv('../morphology_preprocessing/morphology_with_spot.csv', index_col=1)

spot_prediction_df = pd.read_csv('/mnt/Drive E/Class Notes/L-4 T-1/Thesis/ScribbleDom/final_outputs/cancers/bcdc_ffpe/expert/final_barcode_labels.csv', index_col=0)
spot_annotation_df = pd.read_csv('/mnt/Drive E/Class Notes/L-4 T-1/Thesis/ScribbleDom/preprocessed_data/cancers/bcdc_ffpe/manual_annotations.csv', index_col=0)
spot_scribble_df = pd.read_csv('/mnt/Drive E/Class Notes/L-4 T-1/Thesis/ScribbleDom/preprocessed_data/cancers/bcdc_ffpe/manual_scribble.csv', index_col=0)
cell_annotation_df = pd.read_csv('../input/cell_level_annotation.csv', index_col=0)


# Align on index
cell_prediction_df.sort_index(inplace=True)
cell_coord_df.sort_index(inplace=True)
cell_annotation_df.sort_index(inplace=True)

# print("prediction: ", )
# print(cell_prediction_df.iloc[:, -1].value_counts())

cell_coord_df = cell_coord_df[['centroid_x_px', 'centroid_y_px']]

cell_annotation_df = cell_annotation_df[cell_annotation_df.iloc[:, -1] != -1]

common_indices = cell_annotation_df.index.intersection(cell_prediction_df.index)
cell_prediction_df = cell_prediction_df.loc[common_indices]
cell_coord_df = cell_coord_df.loc[common_indices]

cell_prediction_df.iloc[:, -1] = cell_prediction_df.iloc[:, -1] + 1


df_merged = pd.merge(cell_prediction_df, cell_coord_df, left_index=True, right_index=True).dropna()
print("prediction + coord: ", len(df_merged))

plot_scatter(df_merged, f'{output_img_folder}/prediction.png')


# plotting manual annotations

df_merged = pd.merge(cell_annotation_df, cell_coord_df, left_index=True, right_index=True).dropna()
print("annotation + coord: ", len(df_merged))

plot_scatter(df_merged, f'{output_img_folder}/annotation.png')


# print("annotation: ", )
# print(cell_annotation_df.iloc[:, -1].value_counts())
# print("prediction: ", )
# print(cell_prediction_df.iloc[:, -1].value_counts())


# Calculate ARI
def calc_ari(df_1, df_2):

    df_merged = pd.merge(df_1, df_2, left_index=True, right_index=True).dropna()

    cols = df_merged.columns
    # print(f"Columns in merged dataframe: {cols}")
    for col in cols:
        df_merged[col] = df_merged[col].values.astype('int')


    return adjusted_rand_score(df_merged[cols[0]].values, df_merged[cols[1]].values)


ari = calc_ari(cell_annotation_df, cell_prediction_df)

final_output_ari = f'{final_output_folder}/{dataset}/{samples[0]}/morphology/{scheme}/ari.csv'

print(f"Ari for dataset:{dataset} scheme:{scheme} is: ", ari)
pd.DataFrame([{"ARI": ari}]).to_csv(final_output_ari)



# scribble_labels = torch.load("preprocessed/labels.pt", weights_only=True)
# scribble_mask = torch.load("preprocessed/scribble_mask.pt", weights_only=True)
# cell_scribble_df = pd.read_csv("../input/cell_level_scribble.csv", index_col=0)
# cell_prediction_df = pd.read_csv(sell_prediction_file, index_col=0)
#
# print(cell_prediction_df.iloc[:, -1].value_counts())
#
# pred_labels = torch.tensor(cell_prediction_df.values.ravel())
#
# scribble_truth = scribble_labels[scribble_mask]
# pred_scribble = pred_labels[scribble_mask]
#
# matches = (scribble_truth == pred_scribble)
# accuracy = matches.sum().item() / matches.numel()



df_compare = pd.DataFrame({
    'ScribbleLabel': scribble_truth.cpu().numpy(),
    'Prediction': pred_scribble.cpu().numpy()
})

print(matches.numpy().sum())
print(df_compare.iloc[:,0].value_counts())
print(df_compare.iloc[:,1].value_counts())

# print("\ncell scribble: ", cell_scribble_df.iloc[:,-1].value_counts())
# print("\nspot scribble: ", spot_scribble_df.iloc[:,-1].value_counts())
# print("\nspot annotation: ", spot_annotation_df.iloc[:,-1].value_counts())
# print("\ncell annotation: ", cell_annotation_df.iloc[:,-1].value_counts())

# Save to CSV
df_compare.to_csv("scribble_vs_prediction.csv", index=False)

print(f"Scribble Match Accuracy: {accuracy:.4f}")


