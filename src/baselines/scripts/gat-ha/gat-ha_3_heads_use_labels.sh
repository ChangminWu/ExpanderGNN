cd "$(dirname $0)" 
python -u ../../src/main.py \
    --seed 0 \
    --n-label-iters 1 \
    --lr 0.002 \
    --model gat-ha \
    --mode student \
    --alpha 0.9 \
    --temp 0.7 \
    --n-layers 3 \
    --n-hidden 256 \
    --K 3 \
    --n-heads 3 \
    --dropout 0.75 \
    --input_drop 0.25 \
    --edge_drop 0.3 \
    --attn_drop 0. \
    --no-attn-dst \
    --norm sym \
    --n-epochs 2000 \
    --n-runs 10 \
    --use-labels \
    --output-path ../output2/ \
    --save-pred \
    --checkpoint-path ../checkpoint1/ \
    --gpu 0