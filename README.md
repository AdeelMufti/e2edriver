# E2EDriver

This codebase can be used to train a deep learning model on the [Waymo End to End Open Dataset](https://waymo.com/open/data/e2e/).

The model is set up as follows:
* Encode: Images using SqueezeNet (Convolutional) to ViT (Vision Transformer). Positional embedding per surround camera at each tick
* Encode: State: Positional History and Intent using MLPs
* Fusion: Simple fusion of Images, State
* Decode Trajectory: Multiple head formulations
    * Unimodal Regression
    * Multimodal Regression (Best Results)
    * Gaussian Mixtures, as described in http://blog.adeel.io/2019/06/18/context-aware-prediction-with-uncertainty-of-traffic-behaviors-using-mixture-density-networks/

Other model details:
* 12M Params
* AMP - Mixed Precision
* OneCycleLR
* Various image augmentations


## Quickstart
* Setup a Python 3.10+ virtual environment
* Install packages:
```
pip install requirements.txt
pip install requirements_waymo_dataset.txt
```
* Use `data_create.py` to download and preprocess the Waymo Dataset
* Edit configuration in `config.py`, as needed
* Test training locally on CPU
```
python src/main.py --dry --shrink --cpu
```
* Train full model on GPU enabled machine
```
python src/main.py
```

## Cloud Training: AWS
* Setup a docker account on hub.docker.com
* Modify all .sh shell scripts and update $DOCKER_USER with your username
* Upload the preprocessed dataset to S3 bucket
* Create an EC2 machine with a GPU, attach and mount S3 bucket
* Note the EC2 Instance ID
* Run cloud launch script
```
# bash launch_remote.sh <instance id> <experiment-name>
bash launch_remote.sh i-instance123 my_experiment
```
* A `Docker` image will be created and synchronized on the EC2 machine
* Machine will run tensorboard, and shut down after training completes