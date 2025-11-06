#!/bin/bash

#cd ../..

# custom config
export CUDA_VISIBLE_DEVICES=7
export PYTHONPATH=$PYTHONPATH:`pwd`

DATA="./data/"
TRAINER=CoOp

DATASET=$1
CFG=vit_b16_ep100  # config file
CTP=end  # class token position (end or middle)
NCTX=2  # number of context tokens
SHOTS=30  # number of shots (1, 2, 4, 8, 16)
CSC=False  # class-specific context (False or True)

for SEED in 1 2 3
do
    DIR=output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}
    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    TRAINER.COOP.N_CTX ${NCTX} \
    TRAINER.COOP.CSC ${CSC} \
    TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
    DATASET.NUM_SHOTS ${SHOTS}
done