#!/bin/bash

if [ -d "./venv" ]; then
    source ./venv/bin/activate
fi

if [ ! -d "./logs/LTSF" ]; then
    mkdir -p ./logs/LTSF
fi

seq_len=336
model_name=DLinear

root_path_name=./dataset/
data_path_name=ETTm2.csv
model_id_name=ETTm2
data_name=ETTm2

random_seed=2021

for pred_len in 96 192 336 720
do
    case $pred_len in
        96)  learning_rate=0.001;;
        192) learning_rate=0.001;;
        336) learning_rate=0.01;;
        720) learning_rate=0.01;;
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
      --pred_len 96 \
      --enc_in 7 \
      --des 'Exp' \
      --itr 1 \
      --batch_size 32 \
      --learning_rate $learning_rate \
      $@ \
      > >(tee logs/LTSF/${model_name}_${model_id_name}_${seq_len}_${pred_len}.log)  # This will log to both console and file
done