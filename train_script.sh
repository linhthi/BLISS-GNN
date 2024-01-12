#!/bin/bash
K=10
DATASET="flickr"
SAMPLER="poisson-bandit"
MODEL="gat"
BATCH_SIZE=2000
NUM_STEPS=1000
NUM_LAYERS=3
FAN_OUT="4000,8000,10000"
ETA=0.001
GPU=0

for i in $(seq 1 $K)
do
    python train_lightning.py --importance-sampling 1 \
        --dataset $DATASET --num-steps ${NUM_STEPS} \
        --fan-out ${FAN_OUT} \
        --model gat --batch-size ${BATCH_SIZE} --residual \
        --sampler $SAMPLER --num-layers ${NUM_LAYERS} \
        --gpu $GPU --k-runs 1 --eta ${ETA}
done