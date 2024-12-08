python .\train.py ^
    -m "t2t_vit_14" ^
    -d "..\data\tiny-imagenet-200" ^
    -b 256 ^
    -n 200 ^
    -opt "adamw" ^
    -lr 2e-4 ^
    -o "out" ^
    --weight-decay 5e-2 ^
    --lr-scheduler "cosine" ^
    --lr-warmup-epochs 5 ^
    --lr-warmup-method "linear" ^
    --lr-warmup-decay 0.01
    --smoothing 0.1 ^
    --writer