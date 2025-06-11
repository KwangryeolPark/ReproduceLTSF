#!/bin/bash

if [ -d "./venv" ]; then
    source ./venv/bin/activate
fi

if [ ! -d "./logs/LTSF" ]; then
    mkdir -p ./logs/LTSF
fi

rate=0.0001
seq_len=96
model_name=Fredformer

root_path_name=./dataset/
data_path_name=ETTh1.csv
model_id_name=ETTh1
data_name=ETTh1

random_seed=2021

for pred_len in 96 192 336 720
do
    case $pred_len in
        96)  cf_dim=128 cf_depth=2 cf_heads=8 cf_mlp=96 cf_head_dim=32 d_model=24;;
        192) cf_dim=128 cf_depth=2 cf_heads=8 cf_mlp=96 cf_head_dim=32 d_model=24;;
        336) cf_dim=128 cf_depth=2 cf_heads=8 cf_mlp=96 cf_head_dim=32 d_model=24;;
        720) cf_dim=128 cf_depth=2 cf_heads=8 cf_mlp=96 cf_head_dim=32 d_model=24;;
    esac
    python -u run.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --d_model $d_model \
      --d_ff 128 \
      --dropout 0.3 \
      --fc_dropout 0.3 \
      --patch_len 4 \
      --stride 4 \
      --des 'Exp' \
      --train_epochs 100 \
      --patience 10 \
      --itr 1 \
      --batch_size 128 \
      --learning_rate $rate \
      --cf_dim $cf_dim \
      --cf_depth $cf_depth \
      --cf_heads $cf_heads \
      --cf_mlp $cf_mlp \
      --cf_head_dim $cf_head_dim \
      --use_nys 0 \
      --individual 0 \
      $@ \
      > >(tee logs/LTSF/${model_name}_${model_id_name}_${seq_len}_${pred_len}.log)  # This will log to both console and file
done