#!/bin/bash

python main/train_runner_with_reg.py \
--MODEL_VERSION=v6 \
--ALL_DATASET=/all_dataset \
--BATCH_SIZE=128 \
--TEST_SIZE=1000 \
--NUMBER_EPOCHS=200 \
--EMBEDDING_SIZE=200 \
--CONV0_DROPOUT=0.3 \
--CONV1_DROPOUT=0.3 \
--CONV0_NUMBER_FILTERS=5 \
--CONV1_NUMBER_FILTERS=5 \
--CONV0_KERNEL_POOLING_SIZE=2 \
--CONV1_KERNEL_POOLING_SIZE=2 \
--FC0_SIZE=100 \
--CONV0_KERNEL_FILTER_SIZE=5 \
--CONV1_KERNEL_FILTER_SIZE=5 \
--FC0_DROPOUT=0.3 \
--FC1_DROPOUT=0.3 \
--NUM_HIDDEN=100 \
--LEARNING_RATE=0.05 \
--GPU=0.7 \
--REG_RATE=0.1


python main/train_runner_with_reg.py \
--MODEL_VERSION=v7 \
--ALL_DATASET=/all_dataset \
--BATCH_SIZE=128 \
--TEST_SIZE=1000 \
--NUMBER_EPOCHS=200 \
--EMBEDDING_SIZE=200 \
--CONV0_DROPOUT=0.3 \
--CONV1_DROPOUT=0.3 \
--CONV0_NUMBER_FILTERS=5 \
--CONV1_NUMBER_FILTERS=5 \
--CONV0_KERNEL_POOLING_SIZE=2 \
--CONV1_KERNEL_POOLING_SIZE=2 \
--FC0_SIZE=100 \
--CONV0_KERNEL_FILTER_SIZE=5 \
--CONV1_KERNEL_FILTER_SIZE=5 \
--FC0_DROPOUT=0.3 \
--FC1_DROPOUT=0.3 \
--NUM_HIDDEN=100 \
--LEARNING_RATE=0.05 \
--GPU=0.7 \
--REG_RATE=0.0


python main/train_runner_with_reg.py \
--MODEL_VERSION=v7 \
--ALL_DATASET=/all_dataset \
--BATCH_SIZE=128 \
--TEST_SIZE=1000 \
--NUMBER_EPOCHS=200 \
--EMBEDDING_SIZE=200 \
--CONV0_DROPOUT=0.3 \
--CONV1_DROPOUT=0.3 \
--CONV0_NUMBER_FILTERS=5 \
--CONV1_NUMBER_FILTERS=5 \
--CONV0_KERNEL_POOLING_SIZE=2 \
--CONV1_KERNEL_POOLING_SIZE=2 \
--FC0_SIZE=100 \
--CONV0_KERNEL_FILTER_SIZE=5 \
--CONV1_KERNEL_FILTER_SIZE=5 \
--FC0_DROPOUT=0.3 \
--FC1_DROPOUT=0.3 \
--NUM_HIDDEN=100 \
--LEARNING_RATE=0.05 \
--GPU=0.7 \
--REG_RATE=0.1

