#!/bin/sh

nvidia-docker run -d  \
    -v `pwd`:/source/ \
    -v /root/code/all_dataset:/all_dataset \
    trind/sentiment \
    /bin/bash -c "/source/start_predict.sh"
