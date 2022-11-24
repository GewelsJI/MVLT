export NCCL_LL_THRESHOLD=0
EXP_ID='dws_mvlt_ft_exp48'

CUDA_VISIBLE_DEVICES=3,4 ~/miniconda3/envs/MVLT/bin/python3.6 -m torch.distributed.launch --nproc_per_node=2 \
    --master_port=10041 \
    --use_env main_vl.py \
    --config scripts_pt_dws/configs/${EXP_ID}.py \
    --data-path ./Fashion-Gen-Processed \
    --resume checkpoints/${EXP_ID}/checkpoint_recognition.pth \
    --eval-recognition \
    --runtime dws