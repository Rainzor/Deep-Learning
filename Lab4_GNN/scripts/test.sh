echo "-- Testing --"
root = "/data2/wrz/Datasets/"


echo "1. Testing Cora"
python train.py -r $root --dataset cora --tag test -n 50

echo "2. Testing Citeseer"
python train.py -r $root --dataset citeseer --tag test -n 50

echo "3. Testing PPI"
python train.py -r $root --dataset ppi --tag test -n 50

echo "4. Testing GAT"
python train.py -r $root --dataset cora --tag test --model GAT -n 50

