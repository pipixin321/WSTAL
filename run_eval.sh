model_path="/mnt/data1/zhx/Weakly_TAL/LACP_zhx/models/LACP1"

output_path="/mnt/data1/zhx/Weakly_TAL/LACP_zhx/outputs/LACP_eval1"
log_path="/mnt/data1/zhx/Weakly_TAL/LACP_zhx/logs/LACP_eval1" 
model_file="/mnt/data1/zhx/Weakly_TAL/LACP_zhx/models/LACP1/model_seed_1.pkl"

# output_path="/mnt/data1/zhx/Weakly_TAL/LACP_zhx/outputs/LACP_eval_best"
# log_path="/mnt/data1/zhx/Weakly_TAL/LACP_zhx/logs/LACP_eval_best" 
# model_file="/mnt/data1/zhx/Weakly_TAL/LACP_zhx/model_best.pkl"

# output_path="/mnt/data1/zhx/Weakly_TAL/LACP_zhx/outputs/LACP_eval_partloss"
# log_path="/mnt/data1/zhx/Weakly_TAL/LACP_zhx/logs/LACP_eval_partloss" 
# model_file="/mnt/data1/zhx/Weakly_TAL/LACP_zhx/models/LACP/model_seed_0.pkl"

# output_path="/mnt/data1/zhx/Weakly_TAL/LACP_zhx/outputs/LACP_train"
# log_path="/mnt/data1/zhx/Weakly_TAL/LACP_zhx/logs/LACP_train" 
# model_file="/mnt/data1/zhx/Weakly_TAL/LACP_zhx/model_best.pkl"

CUDA_VISIBLE_DEVICES=0 python -W ignore ./main_eval.py --model_path ${model_path} --output_path ${output_path} --log_path ${log_path} --model_file ${model_file}