#!/bin/sh

nvidia-docker run --name trind_tensorboard -d --rm \
-v `pwd`:/source/ \
-p 0.0.0.0:6006:6006 \
tensorflow/tensorflow:nightly-gpu-py3 \
/bin/bash -c "tensorboard --logdir=/source/summary"
