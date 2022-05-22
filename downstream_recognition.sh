export NCCL_LL_THRESHOLD=0
EXP_ID='pai_pvlt_ft_48'

# # >>> runtime: remote-pai
# EXP_ID='pai_pvlt_ft_48'
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 \
    --master_port=10041 \
    --use_env main_vl.py \
    --config scripts_ft_pai/configs/${EXP_ID}.py \
    --data-path /home/admin/workspace/daniel_ji/dataset/Fashion-Gen \
    --resume checkpoints/${EXP_ID}/checkpoint.pth \
    --eval-recognition \
    --runtime dws

# pretraining
# > logging-sup: accuracy (0.9742375799311362) macro_f1 (0.8724180008709451) micro_f1 (0.9742375799311362) weighted_f1 (0.9741176820711007)
# > logging-sub: accuracy (0.9181320708312838) macro_f1 (0.7940458025388675) micro_f1 (0.9181320708312838) weighted_f1 (0.9169663616807412)

# finetune
# > logging-sup: accuracy (0.9825996064928677) macro_f1 (0.8954719842489123) micro_f1 (0.9825996064928677) weighted_f1 (0.9824654977888717)
# > logging-sub: accuracy (0.9356554353172651) macro_f1 (0.8285927576055913) micro_f1 (0.9356554353172651) weighted_f1 (0.9351514388782373)