echo "-- Testing --"
echo "=== Node Classification ==="

# Cora Node Classification: 0.806
python train.py -r /data2/wrz/Datasets/ --dataset cora --task node-cls -lr 0.01 -n 200 --patience 50 --hidden-dim 128 -nl 2 --activation relu --tag best

# Citeseer Node Classification: 0.718
python train.py -r /data2/wrz/Datasets/ --dataset citeseer --task node-cls -lr 0.01 -n 200 --patience 50 --hidden-dim 128 -nl 2 --activation relu --tag best

# PPi Node Classification: 0.7465
python train.py -r /data2/wrz/Datasets/ --dataset ppi --task node-cls -lr 0.02 -n 200 --patience 100 --hidden-dim 256 -nl 4 -pn PN-SI --activation relu --tag best

echo "=== Link Prediction ==="
# Cora Link Prediction: 0.7562
python train.py -r /data2/wrz/Datasets/ --dataset cora --task link-pred -lr 0.01 -n 200 -ws 10 --patience -1 --scheduler cosine --hidden-dim 128 -nl 4  --edge-drop 0.1 --activation relu --tag best

# Citeseer Link Prediction: 0.7681
python train.py -r /data2/wrz/Datasets/ --dataset citeseer --task link-pred -lr 0.01 -n 200 -ws 10 --patience -1 --scheduler cosine --hidden-dim 128 -nl 4 -pn PN-SI --activation relu --tag best

# PPI Link Prediction:  0.6882
python train.py -r /data2/wrz/Datasets/ --dataset ppi --task link-pred -lr 0.02 -n 200 -ws 10 --patience -1 --scheduler cosine --hidden-dim 256 -nl 4  --edge-drop 0.1 --activation relu --tag best