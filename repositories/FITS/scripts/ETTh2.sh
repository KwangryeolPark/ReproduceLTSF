#!/bin/bash

if [ -d "./venv" ]; then
    source ./venv/bin/activate
fi

if [ ! -d "./logs/LTSF" ]; then
    mkdir -p ./logs/LTSF
fi

seq_len=720
model_name=FITS

root_path_name=./dataset/
data_path_name=ETTh2.csv
model_id_name=ETTh2
data_name=ETTh2

random_seed=2021

for pred_len in 96 192 336 720
do
    case "$pred_len" in
        96)
            train_mode=2
            ;;
        192|336|720)
            train_mode=1
            ;;
        *)
            echo "Unsupported pred_len: $pred_len"
            continue
            ;;
    esac

    python run_F.py \
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
        --des 'Exp' \
        --itr 1 \
        --batch_size 64 \
        --learning_rate 0.0005 \
        --train_mode $train_mode \
        --H_order 6 \
        --patience 20 \
        $@ \
        > >(tee logs/LTSF/${model_name}_${model_id_name}_${seq_len}_${pred_len}.log)  # This will log to both console and file
done