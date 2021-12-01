model_path="/mnt/data1/zhx/Weakly_TAL/LACP_zhx/models/LACP"
output_path="/mnt/data1/zhx/Weakly_TAL/LACP_zhx/outputs/LACP"
log_path="/mnt/data1/zhx/Weakly_TAL/LACP_zhx/logs/LACP" 
seed=0

CUDA_VISIBLE_DEVICES=0 python -W ignore ./main.py --model_path ${model_path} --output_path ${output_path} --log_path ${log_path} --seed ${seed}