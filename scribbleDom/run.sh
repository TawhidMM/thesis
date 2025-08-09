#! /bin/bash
config_file_expert="configs/bcdc/bcdc_config_expert.json"
config_file_mclust="configs/bcdc/bcdc_config_mclust.json"

export CUDA_LAUNCH_BLOCKING=1

#python3 graph_constractor.py --params ${config_file_mclust}
#
#python3 train.py --params ${config_file_mclust}
#echo "----- training expert model finished -----"
#
#python3 post_processing.py --params ${config_file_mclust}
#echo "----- post processing expert model finished -----"

python3 show_result.py --params ${config_file_mclust}