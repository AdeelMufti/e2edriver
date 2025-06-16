#!/bin/bash

if [ -z "$1" ]; then
  echo "Usage: $0 <experiment-name>"
  exit 1
fi

mkdir -p /home/adeel/data/logs/$1/
docker exec -it e2edriver bash -c "tensorboard --logdir=/home/adeel/data/logs/$1/ --port 8000 --bind_all"

if [ $? -ne 0 ]; then
  docker container run --rm --ipc host --gpus all --mount type=bind,source=/home/adeel/data,target=/home/adeel/data -p 8000:8000 --name e2edriver -it adeelmufti/e2edriver:v0.1 bash -c "tensorboard --logdir=/home/adeel/data/logs/$1/ --port 8000 --bind_all"
fi