#!/bin/bash

# Default values
n=1
b=512
dropout=0.3
# Parse command line arguments
while getopts "n:b:l:" opt; do
  case $opt in
    n) n=$OPTARG ;;
    b) b=$OPTARG ;;
    d) dropout=$OPTARG ;;
    *) echo "Usage: $0 [-n number_of_layers] [-b batch_size] [-d dropout_rate]" >&2
       exit 1 ;;
  esac
done

echo "-- Testing Basic RNN --"

# echo "1. Testing RNN"
# python train.py -d /data2/wrz/Datasets/Yelp/ -n $n -b $b -lr 1e-4 -m rnn --scheduler cosine --dropout $dropout

echo "2. Testing GRU"
python train.py -d /data2/wrz/Datasets/Yelp/ -n $n -b $b -lr 1e-4 -m gru --scheduler cosine

echo "3. Testing LSTM"
python train.py -d /data2/wrz/Datasets/Yelp/ -n $n -b $b -lr 2e-3 -m lstm --scheduler cosine --dropout $dropout

# echo "4. Testing BiRNN"
# python train.py -d /data2/wrz/Datasets/Yelp/ -n $n -b $b -lr $lr -m rnn --bidirectional --tag bidirect --scheduler cosine --dropout $dropout

echo "5. Testing BiGRU"
python train.py -d /data2/wrz/Datasets/Yelp/ -n $n -b $b -lr 1e-4 -m gru --bidirectional --tag bidirect --scheduler cosine --dropout $dropout

echo "6. Testing BiLSTM"
python train.py -d /data2/wrz/Datasets/Yelp/ -n $n -b $b -lr 2e-3 -m lstm --bidirectional --tag bidirect --scheduler cosine --dropout $dropout
