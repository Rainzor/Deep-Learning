#!/bin/bash

# Default values
n=1
b=512
cycle=2 # number of cycles = 0.5 * number of GPUs
# Parse command line arguments
while getopts "n:b:l:" opt; do
  case $opt in
    n) n=$OPTARG ;;
    b) b=$OPTARG ;;
    *) echo "Usage: $0 [-n number_of_layers] [-b batch_size] [-nc number_of_cycles]" &&
       exit 1 ;;
  esac
done

echo "-- Testing Basic RNN --"

echo "1. Testing RNN"
# 0.569
python train.py -d /data2/wrz/Datasets/Yelp/  -n 50 -b $b -lr 1e-4 -m rnn --min_lr 1e-6 --patience 20 --tag test --num_cycles $cycle --val_ratio 0.05 --dropout 0.5 --weight_decay 1e-5 --smooth 0.1                                                   

echo "--RNN Passed--"

echo "2. Testing GRU"
# 0.625
python train.py -d /data2/wrz/Datasets/Yelp/ -n 20 -b $b -lr 1e-3 -m gru --min_lr 1e-5 --patience 10 --num_cycles $cycle --val_ratio 0.05 --dropout 0.5 --weight_decay 1e-5

echo "--GRU Passed--"

echo "3. Testing LSTM"

# 0.637
python train.py -d /data2/wrz/Datasets/Yelp/ -n 20 -b $b -lr 1e-3 -m lstm --min_lr 1e-5 --patience 10 --num_cycles $cycle --val_ratio 0.05 --dropout 0.5 --weight_decay 1e-5

echo "--LSTM Passed--"

echo "-- Testing Bidirectional --"

# echo "4. Testing BiRNN"
# 0.544
python train.py -d /data2/wrz/Datasets/Yelp/  -n 50 -b $b -lr 2e-4 -m rnn --min_lr 1e-12 --patience 20 --tag test --num_cycles $cycle --val_ratio 0.05 --dropout 0.5 --weight_decay 1e-5 --smooth 0.1 --bidirectional                                           


echo "--BiRNN Passed--"

echo "5. Testing BiGRU"
# 0.645
python train.py -d /data2/wrz/Datasets/Yelp/ -n 20 -b $b -lr 1e-3 -m gru --min_lr 1e-5 --patience 10 --tag bidirect --num_cycles $cycle --val_ratio 0.05 --dropout 0.5 --weight_decay 1e-5 --bidirectional

echo "--BiGRU Passed--"

echo "6. Testing BiLSTM"
# 0.652
python train.py -d /data2/wrz/Datasets/Yelp/ -n 20 -b $b -lr 2e-3 -m lstm --min_lr 1e-5 --patience 10 --tag bidirect --num_cycles $cycle --val_ratio 0.05 --dropout 0.5 --weight_decay 1e-5 --bidirectional

echo "--BiLSTM Passed--"

echo "-- Testing Pooling --"

echo "7. Testing BiLSTM with MaxPooling"
# 0.649
python train.py -d /data2/wrz/Datasets/Yelp/ -n 40 -b $b -lr 2e-3 -m lstm --min_lr 1e-8 --patience 20 --num_cycles $cycle --val_ratio 0.05 --dropout 0.5 --weight_decay 1e-4 --bidirectional --pool max

echo "8. Testing BiLSTM with MeanPooling"
# 0.653
python train.py -d /data2/wrz/Datasets/Yelp/ -n 40 -b $b -lr 3e-3 -m lstm --min_lr 5e-4 --patience 20 --num_cycles $cycle  --val_ratio 0.05 --dropout 0.5 --weight_decay 1e-4 --bidirectional --pool mean

echo "9. Testing BiLSTM with Attention"
