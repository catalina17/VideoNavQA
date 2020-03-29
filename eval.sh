#!/bin/bash

# input args
GPU=${1-0}
mode="${2-film_attn_pt}"  # film_attn_pt, film_gp_pt, time_multi_hop

# model and optimisation args
num_classes=70
vocab_size=134
num_res_blocks=3
num_res_block_channels=1024
num_tail_channels=64
at_hidden_size=128
hidden_size=128
batch_size=32
loss_reduction="sum"
l_rate=0.0001
num_epochs=1
best_acc=0
stats_after_every=500
frcnn_pretrained_path="../vgg16_caffe.pth"

case ${mode} in
time_multi_hop)
  batch_size=16
  l_rate=0.00005
  checkpoint_path="tmh_sum_5e-5_3b_1024f_64t.pt"
  log_file="tmh_sum_5e-5_3b_1024f_64t.log"
  ;;
film_gp_pt)
  num_res_blocks=4
  num_tail_channels=32
  checkpoint_path="gp_sum_1e-4_4b_1024f_32t.pt"
  log_file="gp_sum_1e-4_4b_1024f_32t.log"
  ;;
film_attn_pt)
  num_res_blocks=5
  checkpoint_path="at_sum_1e-4_4b_1024f_128ah_128h.pt"
  log_file="at_sum_1e-4_4b_1024f_128ah_128h.log"
  ;;
esac

cd eval/;
python q_and_v_eval.py \
    --model $mode \
    --num_classes $num_classes \
    --vocab_size $vocab_size \
    --num_res_blocks $num_res_blocks \
    --num_res_block_channels $num_res_block_channels \
    --num_tail_channels $num_tail_channels \
    --at_hidden_size $at_hidden_size \
    --hidden_size $hidden_size \
    --batch_size $batch_size \
    --loss_reduction $loss_reduction \
    --l_rate $l_rate \
    --num_epochs $num_epochs \
    --best_acc $best_acc \
    --frcnn_pretrained_path $frcnn_pretrained_path \
    --checkpoint_path $checkpoint_path \
    --stats_after_every $stats_after_every &>> $log_file
