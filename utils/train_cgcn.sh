#!/bin/bash

SAVE_ID=$1
CUDA_VISIBLE_DEVICES=2 python3 train.py --id $SAVE_ID --seed 0 --prune_k -1 --lr 0.3 --rnn_hidden 200 --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003 --adj_l2 1 --num_layers 4 --data_dir dataset/bc-full_7 --vocab_dir dataset/vocab-bc-full_7
