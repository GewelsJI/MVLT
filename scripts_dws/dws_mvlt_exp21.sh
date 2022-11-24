cd ..
export NCCL_LL_THRESHOLD=0

_CONFIG='dws_pvlt_exp21'
_PORT=10007

mkdir -p ./checkpoints/${_CONFIG}/

cp ./scripts_dws/${_CONFIG}.sh ./scripts_dws/configs/${_CONFIG}.py ./checkpoints/${_CONFIG}/

# parameters settings
# @nproc_per_node: gpus numbers (1 bs -> 70 MB)
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
    --nproc_per_node=1 \
    --master_port=${_PORT} \
    --use_env main_vl.py \
    --config scripts_dws/configs/${_CONFIG}.py \
    --data-path /home/admin/workspace/daniel_ji/dataset/Fashion-Gen \
    --runtime dws