#!/bin/bash

# Default values
n=1
b=256
lr=1e-5

# Parse command line arguments
while getopts "n:b:l:" opt; do
  case $opt in
    n) n=$OPTARG ;;
    b) b=$OPTARG ;;
    l) lr=$OPTARG ;;
    *) echo "Usage: $0 [-n number_of_layers] [-b batch_size] [-l learning_rate]" ;;
  esac
done

echo "-- Testing Basic RNN --"

echo "1. Testing RNN"
python train.py -d /data2/wrz/Datasets/Yelp/ -n $n -b $b -lr $lr -m rnn --scheduler cosine

echo "2. Testing GRU"
python train.py -d /data2/wrz/Datasets/Yelp/ -n $n -b $b -lr $lr -m gru --scheduler cosine

echo "3. Testing LSTM"
python train.py -d /data2/wrz/Datasets/Yelp/ -n $n -b $b -lr $lr -m lstm --scheduler cosine

echo "4. Testing BiRNN"
python train.py -d /data2/wrz/Datasets/Yelp/ -n $n -b $b -lr $lr -m rnn --bidirectional --tag bidirect --scheduler cosine

echo "5. Testing BiGRU"
python train.py -d /data2/wrz/Datasets/Yelp/ -n $n -b $b -lr $lr -m gru --bidirectional --tag bidirect --scheduler cosine

echo "6. Testing BiLSTM"
python train.py -d /data2/wrz/Datasets/Yelp/ -n $n -b $b -lr $lr -m lstm --bidirectional --tag bidirect --scheduler cosine
