#!/bin/bash

python main/train_runner.py \
--ALL_DATASET=/all_dataset \
--BATCH_SIZE=64 \
--TEST_SIZE=2000 \
--NUMBER_EPOCHS=20 \
--EMBEDDING_SIZE=200

python main/train_runner.py \
--ALL_DATASET=/all_dataset \
--BATCH_SIZE=64 \
--TEST_SIZE=2000 \
--NUMBER_EPOCHS=20 \
--EMBEDDING_SIZE=100

python main/train_runner.py \
--ALL_DATASET=/all_dataset \
--BATCH_SIZE=64 \
--TEST_SIZE=2000 \
--NUMBER_EPOCHS=20 \
--EMBEDDING_SIZE=200 \
--CONV0_DROPOUT=0.5 \
--CONV1_DROPOUT=0.5

python main/train_runner.py \
--ALL_DATASET=/all_dataset \
--BATCH_SIZE=64 \
--TEST_SIZE=2000 \
--NUMBER_EPOCHS=20 \
--EMBEDDING_SIZE=200 \
--CONV0_DROPOUT=0.5 \
--CONV1_DROPOUT=0.5 \
--CONV0_NUMBER_FILTERS=20 \
--CONV1_NUMBER_FILTERS=20

python main/train_runner.py \
--ALL_DATASET=/all_dataset \
--BATCH_SIZE=64 \
--TEST_SIZE=2000 \
--NUMBER_EPOCHS=20 \
--EMBEDDING_SIZE=200 \
--CONV0_DROPOUT=0.5 \
--CONV1_DROPOUT=0.5 \
--CONV0_NUMBER_FILTERS=50 \
--CONV1_NUMBER_FILTERS=50


