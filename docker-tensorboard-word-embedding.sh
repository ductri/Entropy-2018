#!/bin/sh

nvidia-docker run --name trind_word_embedding_visualization -d --rm \
-v `pwd`:/source/ \
-p 0.0.0.0:2609:2609 \
tensorflow/tensorflow:nightly-gpu-py3 \
/bin/bash -c "tensorboard --logdir=/source/checkpoint/$1 --port=2609"
