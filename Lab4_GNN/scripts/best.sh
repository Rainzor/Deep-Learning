echo "-- Testing --"
echo "=== Node Classification ==="

# Cora Node Classification: 0.810
python train.py -r /data2/wrz/Datasets/ --dataset cora --task node-cls -lr 0.01 -n 200 --patience 50 --hidden-dim 128 -nl 2 --tag best

# Citeseer Node Classification: 0.7230
python train.py -r /data2/wrz/Datasets/ --dataset citeseer --task node-cls -lr 0.01 -n 200 --patience 50 --hidden-dim 128 -nl 2 --tag best

# PPi Node Classification: 0.7464
python train.py -r /data2/wrz/Datasets/ --dataset ppi --task node-cls -lr 0.02 -n 200 --patience 100 --hidden-dim 256 -nl 4 -pn PN-SI --edge-drop 0.5 --tag best

echo "=== Link Prediction ==="
# Cora Link Prediction: 0.7562
python train.py -r /data2/wrz/Datasets/  --dataset cora --task link-pred -lr 0.01 -n 200 -ws 10 --patience -1 --scheduler cosine --hidden-dim 128 -nl 2 --edge-drop 0.5 -pn PN-SI --tag best

# Citeseer Link Prediction: 0.7681
python train.py -r /data2/wrz/Datasets/  --dataset citeseer --task link-pred -lr 0.01 -n 200 -ws 10 --patience -1 --scheduler cosine --hidden-dim 128 -nl 4 -pn PN-SI --edge-drop 0.5 --tag best

# PPI Link Prediction:  0.6882
python train.py -r /data2/wrz/Datasets/  --dataset ppi --task link-pred -lr 0.02 -n 200 -ws 10 --patience -1 --scheduler cosine --hidden-dim 256 -nl 4 -pn PN-SI --edge-drop 0.5 --tag best