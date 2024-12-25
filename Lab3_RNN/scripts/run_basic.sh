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
python train.py -d /data2/wrz/Datasets/Yelp/  -n 50 -b $b -lr 1e-4 -m rnn --min_lr 1e-6 --patience 20 --num_cycles $cycle --val_ratio 0.05 --dropout 0.5 --weight_decay 1e-5 --smooth 0.1                                                   
echo "0.569"

echo "--RNN Passed--"

echo "2. Testing GRU"
python train.py -d /data2/wrz/Datasets/Yelp/ -n 20 -b $b -lr 1e-3 -m gru --min_lr 1e-5 --patience 10 --num_cycles $cycle --val_ratio 0.05 --dropout 0.5 --weight_decay 1e-5
echo "0.625"

echo "--GRU Passed--"

echo "3. Testing LSTM"

python train.py -d /data2/wrz/Datasets/Yelp/ -n 20 -b $b -lr 1e-3 -m lstm --min_lr 1e-5 --patience 10 --num_cycles $cycle --val_ratio 0.05 --dropout 0.5 --weight_decay 1e-5
echo "0.637"

echo "--LSTM Passed--"

echo "-- Testing Bidirectional --"

# echo "4. Testing BiRNN"
python train.py -d /data2/wrz/Datasets/Yelp/  -n 50 -b $b -lr 2e-4 -m rnn --min_lr 1e-12 --patience 20 --num_cycles $cycle --val_ratio 0.05 --dropout 0.5 --weight_decay 1e-5 --smooth 0.1 --bidirectional --tag bidirect                                           
echo "0.544"

echo "--BiRNN Passed--"

echo "5. Testing BiGRU"
python train.py -d /data2/wrz/Datasets/Yelp/ -n 20 -b $b -lr 1e-3 -m gru --min_lr 1e-5 --patience 10 --tag bidirect --num_cycles $cycle --val_ratio 0.05 --dropout 0.5 --weight_decay 1e-5 --bidirectional
echo "0.645"

echo "--BiGRU Passed--"

echo "6. Testing BiLSTM"
python train.py -d /data2/wrz/Datasets/Yelp/ -n 20 -b $b -lr 2e-3 -m lstm --min_lr 1e-5 --patience 10 --tag bidirect --num_cycles $cycle --val_ratio 0.05 --dropout 0.5 --weight_decay 1e-5 --bidirectional
echo "0.652"

python train.py -d /data2/wrz/Datasets/Yelp/ -n 20 -b 256 -lr 2e-3 -m lstm --patience 10 --tag bidirect-512 --num_cycles $cycle --val_ratio 0.05 --dropout 0.3 --bidirectional --max_length 512 --weight_decay 1e-6
echo "0.661"


python train.py -d /data2/wrz/Datasets/Yelp/ -n 20 -b 256 -lr 5e-3 -m lstm --patience 10 --num_cycles 2 --val_ratio 0.05 --tag bidirect-pretrained --bidirectional --pretrained
echo "0.664"

echo "--BiLSTM Passed--"

echo "-- Testing Pooling --"

echo "7. Testing BiLSTM with MaxPooling"
python train.py -d /data2/wrz/Datasets/Yelp/ -n 40 -b $b -lr 2e-3 -m lstm --min_lr 1e-8 --patience 20 --num_cycles $cycle --val_ratio 0.05 --dropout 0.5 --weight_decay 1e-4 --bidirectional --pool max --tag max
echo "0.649"

python train.py -d /data2/wrz/Datasets/Yelp/ -n 20 -b 256 -lr 4e-3 -m lstm --patience 10 --num_cycles $cycle  --val_ratio 0.05 --max_length 512 --dropout 0.3 --weight_decay 1e-4 --bidirectional --pool max --tag max-512 
echo "0.660"


echo "8. Testing BiLSTM with MeanPooling"
python train.py -d /data2/wrz/Datasets/Yelp/ -n 40 -b $b -lr 3e-3 -m lstm --min_lr 5e-4 --patience 20 --num_cycles $cycle  --val_ratio 0.05 --dropout 0.5 --weight_decay 1e-4 --bidirectional --pool mean --tag mean
echo "0.653"

python train.py -d /data2/wrz/Datasets/Yelp/ -n 40 -b 256 -lr 3e-3 -m lstm --patience 10 --num_cycles $cycle --val_ratio 0.05 --max_length 512 --dropout 0.3 --weight_decay 1e-4 --bidirectional --tag mean-512 --pool mean
echo "0.659"

echo "9. Testing BiLSTM with Attention"
python train.py -d /data2/wrz/Datasets/Yelp/ -n 30 -b $b -lr 5e-3 -m lstm --min_lr 1e-8 --patience 20 --num_cycles $cycle  --val_ratio 0.05 --dropout 0.5 --weight_decay 1e-4 --bidirectional --pool attention --min_warmup 10 --tag attn
echo "0.652"

python train.py -d /data2/wrz/Datasets/Yelp/ -n 20 -b 256 -lr 5e-3 --min_lr 1e-8 --patience 10 -m lstm --num_cycles $cycle  --val_ratio 0.05 --max_length 512 --dropout 0.3 --bidirectional --pool attention --min_warmup 10 --weight_decay 1e-6 --tag attn-512
echo "0.659"