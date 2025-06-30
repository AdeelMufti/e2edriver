#!/bin/bash

DOCKER_USER="adeelmufti"

if [ -z "$1" ]; then
  echo "Usage: $0 <experiment-name> <instance id>"
  exit 1
fi
if [ -z "$2" ]; then
  echo "Usage: $0 <experiment-name> <instance id>"
  exit 1
fi

mkdir -p /home/$USER/data/logs
docker container rm e2edriver
docker container run --rm --ipc host --gpus all --mount type=bind,source=/home/$USER/data,target=/home/$USER/data -p 8000:8000 --name e2edriver -it $DOCKER_USER/e2edriver:v0.1 bash -c "python src/main.py --experiment-name $1 > /home/$USER/data/logs/$1.log 2>&1 || true; exit" || true
aws ec2 stop-instances --instance-ids $2