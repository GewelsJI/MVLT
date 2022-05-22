export NCCL_LL_THRESHOLD=0

# >>> runtime: local-dws
# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=1 \
#     --master_port=10014 \
#     --use_env main_vl.py \
#     --config scripts_dws/configs/dws_exp10_pvlt_tiny.py \
#     --data-path /home/admin/workspace/daniel_ji/dataset/Fashion-Gen \
#     --resume checkpoints/dws_exp10_pvlt_tiny/checkpoint.pth \
#     --eval \
#     --runtime dws

# >>> runtime: remote-pai
_CONFIG='pai_pvlt_exp21'
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 \
    --master_port=10014 \
    --use_env main_vl.py \
    --config scripts_pt_pai/configs/${_CONFIG}.py \
    --data-path /home/admin/workspace/daniel_ji/dataset/Fashion-Gen \
    --resume checkpoints/${_CONFIG}/checkpoint.pth \
    --eval \
    --runtime dws

# ** mlm@acc 0.00000 i2t@acc 0.19386 itm@acc 0.97255 itg@psnr 0.00000 t2i@psnr 57.79511 bartMSS@acc 0.00000 loss 43.45520




########## only for visualization ##########

# >>> runtime: remote-pai
# only support bs=1
# CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 \
#     --master_port=10024 \
#     --use_env main_vl.py \
#     --config scripts_pt_pai/configs/pai_pvlt_exp21.py \
#     --data-path /home/admin/workspace/daniel_ji/dataset/Fashion-Gen \
#     --resume checkpoints/pai_pvlt_exp21/checkpoint.pth \
#     --viz \
#     --runtime dws

# CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 \
#     --master_port=10024 \
#     --use_env main_vl.py \
#     --config scripts_pt_pai/configs/pai_pvlt_exp21.py \
#     --data-path /home/admin/workspace/daniel_ji/workspace/alibaba-pvlt-daniel/alibaba_data \
#     --resume checkpoints/pai_pvlt_exp21/checkpoint.pth \
#     --viz \
#     --runtime dws

# >>> runtime: demo for evaluation
# only support bs=1
# CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 \
#     --master_port=10014 \
#     --use_env main_vl.py \
#     --config scripts_pai/configs/pai_pvlt_exp4.py \
#     --data-path /home/admin/workspace/daniel_ji/workspace/alibaba-pvlt-daniel/dataset_for_clean \
#     --resume checkpoints/pai_pvlt_exp4/checkpoint.pth \
#     --eval \
#     --runtime dws