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

# PPi Node Classification: 0.7489
python train.py -r /data2/wrz/Datasets/ --dataset ppi --task node-cls -lr 0.01 -n 200 --patience 100 --hidden-dim 64 -nl 2

echo "=== Link Prediction ==="
# Cora Link Prediction: 0.697
python train.py -r $root --dataset cora --task link-pred -lr 0.03 -n 200 -ws 10 --patience 100 --scheduler cosine --hidden-dim 256 -nl 3

# Citeseer Link Prediction: 0.713
python train.py -r $root --dataset citeseer --task link-pred -lr 0.03 -n 200 -ws 10 --patience 100 --scheduler cosine --hidden-dim 512 -nl 3

# PPI Link Prediction:  0.699
python train.py -r $root --dataset ppi --task link-pred -lr 0.03 -n 200 -ws 10 --patience 100 --scheduler cosine --hidden-dim 512 -nl 3