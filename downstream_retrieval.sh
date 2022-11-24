# export NCCL_LL_THRESHOLD=0
EXP_ID='dsw_mvlt_exp21'

CUDA_VISIBLE_DEVICES=3,4 ~/miniconda3/envs/MVLT/bin/python3.6 -m torch.distributed.launch --nproc_per_node=2 \
    --master_port=10035 \
    --use_env main_vl.py \
    --config scripts_dws/configs/${EXP_ID}.py \
    --data-path /data/users/jigepeng/Dataset/Fashion-Gen-Processed \
    --resume checkpoints/${EXP_ID}/checkpoint_retrieval.pth \
    --eval-retrieval-itr \
    --runtime dws

CUDA_VISIBLE_DEVICES=3,4 ~/miniconda3/envs/MVLT/bin/python3.6 -m torch.distributed.launch --nproc_per_node=2 \
    --master_port=10037 \
    --use_env main_vl.py \
    --config scripts_dws/configs/${EXP_ID}.py \
    --data-path /data/users/jigepeng/Dataset/Fashion-Gen-Processed \
    --resume checkpoints/${EXP_ID}/checkpoint_retrieval.pth \
    --eval-retrieval-tir \
    --runtime dws