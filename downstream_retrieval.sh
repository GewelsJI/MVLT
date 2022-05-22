# export NCCL_LL_THRESHOLD=0
EXP_ID='pai_mvlt_exp21'

# >>> runtime: remote-pai
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 \
    --master_port=10035 \
    --use_env main_vl.py \
    --config scripts_pt_dws/configs/${EXP_ID}.py \
    --data-path /home/admin/workspace/daniel_ji/dataset/Fashion-Gen \
    --resume checkpoints/${EXP_ID}/checkpoint.pth \
    --eval-retrieval-itr \
    --runtime dws

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 \
    --master_port=10037 \
    --use_env main_vl.py \
    --config scripts_pt_dws/configs/${EXP_ID}.py \
    --data-path /home/admin/workspace/daniel_ji/dataset/Fashion-Gen \
    --resume checkpoints/${EXP_ID}/checkpoint.pth \
    --eval-retrieval-tir \
    --runtime dws