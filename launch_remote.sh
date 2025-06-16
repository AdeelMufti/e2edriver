if [ -z "$1" ]; then
  echo "Usage: $0 <instance id> <experiment-name>"
  exit 1
fi

if [ -z "$2" ]; then
  echo "Usage: $0 <instance id> <experiment-name>"
  exit 1
fi

instance_id=$1
experiment_name=$2
# random_number=$(( $RANDOM % 1000 + 1 ))

echo "Building Docker image..."
# sudo docker system prune -a -f
# sudo docker image rm -f e2edriver:v0.1
# sudo docker build -t e2edriver:v0.1 .
# sudo docker save e2edriver:v0.1 > ~/e2edriver-docker-$random_number.tar
# pigz -c -9 ~/e2edriver-docker-$random_number.tar > ~/e2edriver-docker-$random_number.tar.gz
# echo "Docker image saved and compressed as e2edriver-docker-$random_number.tar.gz"
sudo docker build -t adeelmufti/e2edriver:v0.1 .
sudo docker push adeelmufti/e2edriver:v0.1
echo "Docker image uploaded"

echo "Starting instance..."
aws ec2 start-instances --instance-ids $instance_id
echo "Waiting for instance to start..."
sleep 90
aws ec2 wait instance-running --instance-ids $instance_id
ip_address=$(aws ec2 describe-instances --instance-ids $instance_id --query "Reservations[0].Instances[0].PublicIpAddress" --output text)
echo "Instance started with IP address: $ip_address"
echo "To ssh into the instance, run: ssh -i ~/.ssh/aws_key adeel@$ip_address"

# echo "Copying Docker image to instance..."
# scp -i ~/.ssh/aws_key ~/e2edriver-docker-$random_number.tar.gz adeel@$ip_address:.
# ssh -i ~/.ssh/aws_key adeel@$ip_address "gunzip e2edriver-docker-$random_number.tar.gz"
# ssh -i ~/.ssh/aws_key adeel@$ip_address "docker image rm -f e2edriver:v0.1"
# ssh -i ~/.ssh/aws_key adeel@$ip_address "docker system prune -a -f"
# ssh -i ~/.ssh/aws_key adeel@$ip_address "docker load --input e2edriver-docker-$random_number.tar"
# ssh -i ~/.ssh/aws_key adeel@$ip_address "rm e2edriver-docker-$random_number.tar"
# rm ~/e2edriver-docker-$random_number.tar
# rm ~/e2edriver-docker-$random_number.tar.gz
echo "Updating docker remotely"
ssh -i ~/.ssh/aws_key adeel@$ip_address "docker pull adeelmufti/e2edriver:v0.1"

echo "Running Docker container and training job on instance in screen..."
scp -i ~/.ssh/aws_key launch_local.sh adeel@$ip_address:.
ssh -i ~/.ssh/aws_key adeel@$ip_address "screen -m -d bash launch_local.sh $experiment_name $instance_id"
sleep 60
echo "Training job started."

echo "Running tensorboard in Docker container on instance in screen..."
scp -i ~/.ssh/aws_key launch_tensorboard.sh adeel@$ip_address:.
ssh -i ~/.ssh/aws_key adeel@$ip_address "screen -m -d bash launch_tensorboard.sh $experiment_name"
echo "Tensorboard is running on the instance. You can access it at http://$ip_address:8000"

echo "Tailing logs on instance."
echo ssh -i ~/.ssh/aws_key adeel@$ip_address "tail -f -n 1000 /home/adeel/data/logs/$experiment_name.log"
ssh -i ~/.ssh/aws_key adeel@$ip_address "tail -f -n 1000 /home/adeel/data/logs/$experiment_name.log"