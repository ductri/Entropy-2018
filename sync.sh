#!/bin/sh


echo "Sync all .."
scp ./main/*.py root@213.246.38.101:/root/code/trind/entropy_2018/main
scp ./main/*.yaml root@213.246.38.101:/root/code/trind/entropy_2018/main
scp ./Dockerfile root@213.246.38.101:/root/code/trind/entropy_2018
scp ./docker-run.sh root@213.246.38.101:/root/code/trind/entropy_2018
scp ./docker-build.sh root@213.246.38.101:/root/code/trind/entropy_2018
