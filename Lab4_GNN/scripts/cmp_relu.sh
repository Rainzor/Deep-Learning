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

python train.py -r /data2/wrz/Datasets/ --dataset cora --task node-cls -lr 0.01 -n 200 --patience 50 --hidden-dim 128 -nl 2 --activation relu --tag relu

python train.py -r /data2/wrz/Datasets/ --dataset citeseer --task node-cls -lr 0.01 -n 200 --patience 50 --hidden-dim 128 -nl 2 --activation relu --tag relu

python train.py -r /data2/wrz/Datasets/ --dataset ppi --task node-cls -lr 0.02 -n 200 --patience 100 --hidden-dim 256 -nl 2 --activation relu --tag relu

echo "=== Link Prediction ==="
python train.py -r $root --dataset cora --task link-pred -lr 0.01 -n 200 -ws 10 --patience -1 --scheduler cosine --hidden-dim 128 -nl 2 --activation relu --tag relu

python train.py -r $root --dataset citeseer --task link-pred -lr 0.01 -n 200 -ws 10 --patience -1 --scheduler cosine --hidden-dim 128 -nl 2 --activation relu --tag relu

python train.py -r $root --dataset ppi --task link-pred -lr 0.02 -n 200 -ws 10 --patience -1 --scheduler cosine --hidden-dim 256 -nl 2 --activation relu --tag relu