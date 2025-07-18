#!/bin/bash

if [ -d "./venv" ]; then
    source ./venv/bin/activate
fi

if [ ! -d "./logs/LTSF" ]; then
    mkdir -p ./logs/LTSF
fi

seq_len=96
model_name=Autoformer

root_path_name=./dataset/
data_path_name=exchange_rate.csv
model_id_name=Exchange
data_name=custom

random_seed=2021
for pred_len in 96 192 336 720
do
    case $pred_len in
        96) train_epochs=10;;
        192) train_epochs=1;;
        336) train_epochs=1;;
        720) train_epochs=10;;
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
      --label_len 48 \
      --pred_len $pred_len \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 8 \
      --dec_in 8 \
      --c_out 8 \
      --des 'Exp' \
      --itr 1 \
      --train_epochs $train_epochs \
      $@ \
      > >(tee logs/LTSF/${model_name}_${model_id_name}_${seq_len}_${pred_len}.log)  # This will log to both console and file
done