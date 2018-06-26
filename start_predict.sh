#!/bin/bash

python main/predict_sample.py \
--ALL_DATASET=/all_dataset \
--BATCH_SIZE=1024 \
--EXP_NAME=2018-06-25T13:48:53 \
--STEP=6000 \
--OUTPUT=/source/result.csv
