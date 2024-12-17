#!/usr/bin bash
SEED=0
DATASET='banking'
KNOWN_CLS_RATIO=0.75

python train.py \
    --dataset $DATASET \
    --gpu_id 0 \
    --known_cls_ratio $KNOWN_CLS_RATIO \
    --seed $SEED \
    --freeze_bert_parameters \
    --pretrain \
    --save_model True