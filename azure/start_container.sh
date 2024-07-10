#!/bin/bash

name=${1-deepscale}
image=deepscale/deepscale:latest
echo "starting docker image named $name"
docker run -d -t --name $name \
        --network host \
        -v ${HOME}/workdir:/home/deepscale/workdir \
        -v ${HOME}/.ssh:/home/deepscale/.ssh \
        -v /job/hostfile:/job/hostfile \
        --gpus all $image bash -c 'sudo service ssh start && sleep infinity'
