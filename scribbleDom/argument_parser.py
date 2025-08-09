import argparse
import json

parser = argparse.ArgumentParser(description='ScribbleSeg expert annotation pipeline')
parser.add_argument('--params', help="The input parameters json file path", required=True)

args = parser.parse_args()

with open(args.params) as f:
   params = json.load(f)

dataset = params['dataset']
n_pcs = params['n_pcs']
max_iter = params['max_iter']
nConv = params['nConv']
seed_options = params['seed_options']
lr_options = params['lr_options']
alpha_options = params['alpha_options']
beta_options = params['beta_options']
samples = params['samples']
n_cluster = params['n_cluster_for_auto_scribble']
matrix_format_representation_of_data_path = params['matrix_represenation_of_ST_data_folder']
output_data_path = params['model_output_folder']
final_output_folder = params['final_output_folder']
scheme = params['schema']