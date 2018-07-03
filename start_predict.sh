#!/bin/bash

python main/predict.py \
--ALL_DATASET=/all_dataset \
--BATCH_SIZE=512 \
--EXP_NAME=2018-06-26T16:25:00 \
--STEP=73600 \
--INPUT_FILE=input.csv \
--OUTPUT_FILE=output.csv
