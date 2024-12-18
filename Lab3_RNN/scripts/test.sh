echo "-- Testing --"

echo "Testing RNN"
python train.py -d /data2/wrz/Datasets/Yelp/ -n 1 -b 256 -lr 1e-7 -m rnn

echo "Testing LSTM"
python train.py -d /data2/wrz/Datasets/Yelp/ -n 1 -b 256 -lr 1e-7 -m lstm

echo "Testing GRU"
python train.py -d /data2/wrz/Datasets/Yelp/ -n 1 -b 256 -lr 1e-7 -m gru

echo "Testing BiLSTM"
python train.py -d /data2/wrz/Datasets/Yelp/ -n 1 -b 256 -lr 1e-7 -m lstm --bidirectional

echo "Testing RCNN"
python train.py -d /data2/wrz/Datasets/Yelp/ -n 1 -b 256 -lr 1e-7 -m lstm --pool max

echo "Testing LSTM Attention"
python train.py -d /data2/wrz/Datasets/Yelp/ -n 1 -b 256 -lr 1e-7 -m lstm --pool attention

echo "Testing Transformer"
python train.py -d /data2/wrz/Datasets/Yelp/ -n 1 -b 256 -lr 1e-7 -m transformer

echo "--Testing Custom--"

echo "Testing Custom CNN"
python train.py -d /data2/wrz/Datasets/Yelp/ -n 1 -b 256 -lr 1e-7 -m custom_rnn

echo "Testing Custom LSTM"
python train.py -d /data2/wrz/Datasets/Yelp/ -n 1 -b 256 -lr 1e-7 -m custom_lstm

echo "Testing Custom GRU"
python train.py -d /data2/wrz/Datasets/Yelp/ -n 1 -b 256 -lr 1e-7 -m custom_gru