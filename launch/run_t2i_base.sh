#!/usr/bin/bash

# Text-to-image training, base model (single node, all visible GPUs).
# Run from anywhere: the script moves to the repository root.
cd "$(dirname "$0")/.." || exit
export OMP_NUM_THREADS=4

# Path
data_folder="path/to/t2i_features/"             # contains GPIC/, Fine-T2I/, RenderedText/ shards
vit_folder="path/to/saved_network/t2i_base/"
writer_log="path/to/logs/t2i_base/"
vqgan_folder="path/to/vq_ds16_t2i.pt"
data="gpic+fine_t2i+rendered_text"

# Model size and compute FLOP
vit_size="base"
img_size=256
f_factor=16
codebook_size=16384
register=1
proj=1
dtype="bfloat16"
nb_class=3

# Dataloader
num_workers=8
global_bsize=128
gradacc=1

# Learning hyper-parameter
max_iter=400000
warm_up=2500
lr=2e-4
grad_clip=1
sched_mode="arccos"
dropout=0.0

# sampling
sampler="halton"
sm_temp=1.0
cfg_w=3            # (visualization or test-only)
step=64

# halton scheduler only
top_k=-1
top_p=0.8
temp_warmup=0
sm_temp_min=1.
log_iter=1000

# Corruption of the visible context while training (shuffle, p=0.2 was the previously hard-coded default)
p_resample=0.2
resample_method="shuffle"   # shuffle|local_swap|freq|codebook_nn|self_sample

torchrun --standalone --nnodes=1 --nproc_per_node=gpu -m bigfix.main   --data-folder "${data_folder}" --vit-folder "${vit_folder}" --vqgan-folder "${vqgan_folder}"   --writer-log "${writer_log}" --num-workers ${num_workers} --mode "txt-to-img" --lr "${lr}" --grad-cum "${gradacc}"   --global-bsize "${global_bsize}" --warm-up "${warm_up}" --img-size "${img_size}" --vit-size "${vit_size}"   --data "${data}" --dtype "${dtype}" --proj "${proj}"   --f-factor "${f_factor}" --codebook-size "${codebook_size}" --mask-value "${codebook_size}"   --max-iter "${max_iter}" --sm-temp-min "${sm_temp_min}" --sm-temp "${sm_temp}" --cfg-w "${cfg_w}"   --temp-warmup "${temp_warmup}" --step "${step}" --top-k "${top_k}"   --grad-clip "${grad_clip}" --sched-mode "${sched_mode}" --sampler "${sampler}" --nb-class "${nb_class}"   --register "${register}" --dropout "${dropout}" --top-p "${top_p}" --log-iter "${log_iter}"   --p-resample "${p_resample}" --resample-method "${resample_method}"   --resume
