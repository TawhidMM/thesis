#! /bin/bash
config_file_expert="configs/bcdc/bcdc_config_expert.json"
config_file_mclust="/configs/bcdc/bcdc_config_mclust.json"


#python3 graph_constractor.py
#
#python3 train.py --params ${config_file_expert}
#echo "----- training expert model finished -----"
#
#python3 post_processing.py --params ${config_file_expert}
#echo "----- post processing expert model finished -----"

python3 show_result.py --params ${config_file_expert}