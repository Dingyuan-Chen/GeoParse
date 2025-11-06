#!/bin/bash

#cd ../..

# custom config
export CUDA_VISIBLE_DEVICES=4
export PYTHONPATH=$PYTHONPATH:`pwd`

DATA="./data/"
TRAINER=ZeroshotCLIP
DATASET=$1
CFG=vit_b16_ep100  # rn50, rn101, vit_b32 or vit_b16

python train.py \
--root ${DATA} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/CoOp/${CFG}.yaml \
--output-dir output/${TRAINER}/${CFG}/${DATASET} \
--eval-only