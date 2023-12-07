#!/bin/bash

cd ..

# custom config
DATA=/data/yht/data/cl/data/
TRAINER=KgCoOp
WEIGHT=8.0
CFG=vit_b16_ep100_ctxv1 
CTP=end  # class token position (end or middle)
NCTX=4  # number of context tokens
SHOTS=4  # number of shots 
CSC=False  # class-specific context (False or True)

for SHOTS in 4
do
for DATASET in dtd eurosat fgvc_aircraft food101  oxford_flowers oxford_pets stanford_cars ucf101 caltech101 sun397 imagenet
do
for SEED in 1 2 3
do
    DIR=output_1120_xd/base2new/train_base/${DATASET}/shots_${SHOTS}_${WEIGHT}/${TRAINER}/${CFG}/seed${SEED}
    if [ -d "$DIR" ]; then
        echo "Results are available in ${DIR}. Skip this job"
    else
        echo "Run this job and save the output to ${DIR}"
        CUDA_VISIBLE_DEVICES=7 python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        TRAINER.COOP.N_CTX ${NCTX} \
        TRAINER.COOP.CSC ${CSC} \
        TRAINER.COOP.W ${WEIGHT} \
        TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
        DATASET.NUM_SHOTS ${SHOTS}
    fi
done
done
done
