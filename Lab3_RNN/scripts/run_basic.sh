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

python train.py -d /data2/wrz/Datasets/Yelp/ -n 250 -b 512 -lr 1e-4 -m rnn --scheduler cosine --min_lr 5e-6 --patience 50 --num_cycles $cycle

echo "--RNN Passed--"

echo "2. Testing GRU"
python train.py -d /data2/wrz/Datasets/Yelp/ -n 150 -b 512 -lr 1e-3 -m gru --scheduler cosine --min_lr 1e-4 --patience 30 --num_cycles $cycle

echo "--GRU Passed--"

echo "3. Testing LSTM"
python train.py -d /data2/wrz/Datasets/Yelp/ -n 100 -b 512 -lr 1e-3 -m lstm --scheduler cosine --min_lr 1e-4 --patience 20 --num_cycles $cycle

echo "--LSTM Passed--"

echo "-- Testing Bidirectional --"

# echo "4. Testing BiRNN"
python train.py -d /data2/wrz/Datasets/Yelp/ -n 250 -b 512 -lr 1e-4 -m rnn --scheduler cosine --min_lr 5e-6 --patience 50 --bidirectional --tag bidirect --num_cycles $cycle

echo "--BiRNN Passed--"

echo "5. Testing BiGRU"
python train.py -d /data2/wrz/Datasets/Yelp/ -n 150 -b 512 -lr 1e-3 -m gru --scheduler cosine --min_lr 1e-4 --patience 30 --bidirectional --tag bidirect --num_cycles $cycle

echo "--BiGRU Passed--"

echo "6. Testing BiLSTM"
python train.py -d /data2/wrz/Datasets/Yelp/ -n 100 -b 512 -lr 1e-3 -m lstm --scheduler cosine --min_lr 1e-4 --patience 20 --bidirectional --tag bidirect --num_cycles $cycle

echo "--BiLSTM Passed--"