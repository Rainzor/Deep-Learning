echo "-- Testing --"
root="/data2/wrz/Datasets/"
n=50
b=1
while getopts "n:b:l:" opt; do
  case $opt in
    n) n=$OPTARG ;;
    b) b=$OPTARG ;;
    r) root=$OPTARG ;;
    *) echo "Usage: $0 [-n number_of_layers] [-b batch_size] [-r root]" &&
       exit 1 ;;
  esac
done

echo "=== Node Classification ==="

# Cora Node Classification: 0.810
python train.py -r /data2/wrz/Datasets/ --dataset cora --task node-cls -lr 0.01 -n 200 --patience 50 --hidden-dim 128 -nl 4  --edge-drop 0.5 --tag l4-ed-relu --activation relu

# Citeseer Node Classification: 0.7230
python train.py -r /data2/wrz/Datasets/ --dataset citeseer --task node-cls -lr 0.01 -n 200 --patience 50 --hidden-dim 128 -nl 4  --edge-drop 0.5 --tag l4-ed-relu --activation relu

# PPi Node Classification: 0.7438
python train.py -r /data2/wrz/Datasets/ --dataset ppi --task node-cls -lr 0.02 -n 200 --patience 100 --hidden-dim 256 -nl 4  --edge-drop 0.5 --tag l4-ed-relu --activation relu

echo "=== Link Prediction ==="
# Cora Link Prediction: 0.6973
python train.py -r $root --dataset cora --task link-pred -lr 0.01 -n 200 -ws 10 --patience -1 --scheduler cosine --hidden-dim 128 -nl 4  --edge-drop 0.5 --tag l4-ed-relu --activation relu

# Citeseer Link Prediction: 0.7077
python train.py -r $root --dataset citeseer --task link-pred -lr 0.01 -n 200 -ws 10 --patience -1 --scheduler cosine --hidden-dim 128 -nl 4  --edge-drop 0.5 --tag l4-ed-relu --activation relu

# PPI Link Prediction:  0.6598
python train.py -r $root --dataset ppi --task link-pred -lr 0.02 -n 200 -ws 10 --patience -1 --scheduler cosine --hidden-dim 256 -nl 4  --edge-drop 0.5 --tag l4-ed-relu --activation relu