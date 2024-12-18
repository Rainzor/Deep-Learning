echo "-- Testing --"

echo "1. Testing RNN"
python train.py -d /data2/wrz/Datasets/Yelp/ -n 1 -b 256 -lr 1e-7 -m rnn

echo "2. Testing LSTM"
python train.py -d /data2/wrz/Datasets/Yelp/ -n 1 -b 256 -lr 1e-7 -m lstm

echo "3. Testing GRU"
python train.py -d /data2/wrz/Datasets/Yelp/ -n 1 -b 256 -lr 1e-7 -m gru

echo "4. Testing BiLSTM"
python train.py -d /data2/wrz/Datasets/Yelp/ -n 1 -b 256 -lr 1e-7 -m lstm --bidirectional

echo "5. Testing RCNN"
python train.py -d /data2/wrz/Datasets/Yelp/ -n 1 -b 256 -lr 1e-7 -m lstm --pool max

echo "6. Testing LSTM Attention"
python train.py -d /data2/wrz/Datasets/Yelp/ -n 1 -b 256 -lr 1e-7 -m lstm --pool attention

echo "7. Testing Transformer"
python train.py -d /data2/wrz/Datasets/Yelp/ -n 1 -b 256 -lr 1e-7 -m transformer

echo "--Testing Custom--"

echo "8. Testing Custom CNN"
python train.py -d /data2/wrz/Datasets/Yelp/ -n 1 -b 256 -lr 1e-7 -m custom_rnn

echo "9. Testing Custom LSTM"
python train.py -d /data2/wrz/Datasets/Yelp/ -n 1 -b 256 -lr 1e-7 -m custom_lstm

echo "10. Testing Custom GRU"
python train.py -d /data2/wrz/Datasets/Yelp/ -n 1 -b 256 -lr 1e-7 -m custom_gru