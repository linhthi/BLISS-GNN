#!/bin/bash
for dataset in "cora" "citeseer"
do
    for seed in 12 32 86 34 56 113 1240 88 100 10
    do
        for sampler in "bandit" "ladies" "poisson-bandit"
        do
            python train_lightning.py --importance-sampling 1 \
                --dataset $dataset --num-steps 200 \
                --fan-out 10,10,10 \
                --model gat --batch-size 32 --residual \
                --sampler $sampler --num-layers 3 \
                --gpu 0 --allow-zero-in-degree \
                --seed $seed
        done
    done
done
            