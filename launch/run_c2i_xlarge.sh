#!/usr/bin/bash

# Class-to-image training, xlarge model (single node, all visible GPUs).
# Run from anywhere: the script moves to the repository root.
cd "$(dirname "$0")/.." || exit
export OMP_NUM_THREADS=4

# Path
data_folder="path/to/imagenet_features/"        # output of scripts/extract_vq_features.py
eval_folder="path/to/ImageNet/"                 # raw ImageNet, used for FID evaluation
vit_folder="path/to/saved_network/c2i_xlarge/"
writer_log="path/to/logs/c2i_xlarge/"
vqgan_folder="path/to/vq_ds16_c2i.pt"
data="imagenet_feat"

# Model size and compute FLOP
vit_size="xlarge"
img_size=384
f_factor=16
codebook_size=16384
register=1
proj=1
dtype="bfloat16"

# Dataloader
num_workers=8
global_bsize=256
gradacc=4
nb_class=1000

# Learning hyper-parameter
max_iter=1000000
warm_up=2500
lr=1e-4
grad_clip=1
sched_mode="arccos"
dropout=0.1

# sampling
sampler="halton"
sm_temp=1.0
cfg_w=1.5            # (visualization or test-only)
step=32

# halton scheduler only
top_k=-1
temp_warmup=1
sm_temp_min=1.

# Confidence Sampler only
r_temp=5

torchrun --standalone --nnodes=1 --nproc_per_node=gpu -m bigfix.main   --data-folder "${data_folder}" --vit-folder "${vit_folder}" --vqgan-folder "${vqgan_folder}"   --writer-log "${writer_log}" --num-workers ${num_workers} --mode "cls-to-img" --lr "${lr}" --grad-cum "${gradacc}"   --global-bsize "${global_bsize}" --warm-up "${warm_up}" --img-size "${img_size}" --vit-size "${vit_size}"   --data "${data}" --dtype "${dtype}" --eval-folder "${eval_folder}" --proj "${proj}"   --f-factor "${f_factor}" --codebook-size "${codebook_size}" --mask-value "${codebook_size}"   --max-iter "${max_iter}" --sm-temp-min "${sm_temp_min}" --sm-temp "${sm_temp}" --cfg-w "${cfg_w}"   --temp-warmup "${temp_warmup}" --step "${step}" --top-k "${top_k}"   --grad-clip "${grad_clip}" --sched-mode "${sched_mode}" --sampler "${sampler}" --r-temp "${r_temp}"   --register "${register}" --dropout "${dropout}" --nb-class "${nb_class}"   --resume --compile
