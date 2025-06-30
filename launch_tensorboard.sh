#!/bin/bash

$DOCKER_USER="adeelmufti"

if [ -z "$1" ]; then
  echo "Usage: $0 <experiment-name>"
  exit 1
fi

mkdir -p /home/$USER/data/logs/$1/
docker exec -it e2edriver bash -c "tensorboard --logdir=/home/$USER/data/logs/$1/ --port 8000 --bind_all"

if [ $? -ne 0 ]; then
  docker container run --rm --ipc host --gpus all --mount type=bind,source=/home/$USER/data,target=/home/$USER/data -p 8000:8000 --name e2edriver -it $DOCKER_USER/e2edriver:v0.1 bash -c "tensorboard --logdir=/home/$USER/data/logs/$1/ --port 8000 --bind_all"
fi