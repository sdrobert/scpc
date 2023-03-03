#! /usr/bin/bash

exit 0
# the below aren't meant to be run exactly as-is. I didn't run them this way.
# However, it should give you a good idea of what to do.
#
# Follows along with
# https://aws.amazon.com/blogs/machine-learning/train-deep-learning-models-on-gpus-using-amazon-ec2-spot-instances/

SECURITY_GROUP_ID=  # make sure this is permissive enough to ssh into publicly
KEY_NAME=  # your ssh key
SUBNET_ID=
VOL_SIZE=200
AWS_REGION=ca-central-1
CPU_INSTANCE_TYPE=m4.xlarge
IMAGE_ID=$(aws ec2 describe-images --region $AWS_REGION --owners amazon --filters 'Name=name,Values=Deep Learning AMI GPU PyTorch 1.13.? (Ubuntu 20.04) ????????' 'Name=state,Values=available' --query 'reverse(sort_by(Images, &CreationDate))[:1].ImageId' --output text)

instance_id=$(aws ec2 run-instances --image-id $IMAGE_ID --count 1 --security-group-ids "$SECURITY_GROUP_ID" --instance-type "$CPU_INSTANCE_TYPE" --key-name "$KEY_NAME" --query "Instances[0].InstanceId" --output text)
instance_az=$(aws ec2 describe-instances --query "Reservations[].Instances[?InstanceId=='$instance_id'].Placement.AvailabilityZone" --output text)
volume_id=$(aws ec2 create-volume --size $VOL_SIZE --region $AWS_REGION --availability-zone $instance_az --volume-type gp2 --tag-specifications 'ResourceType=volume,Tags=[{Key=Name,Value=scpc-artifacts}]' --query "VolumeId" --output text)
aws ec2 attach-volume --volume-id $volume_id --instance-id $instance_id --device /dev/sdf
public_dns_name=$(aws ec2 describe-instances --query "Reservations[].Instances[?InstanceId=='$instance_id'].PublicDnsName" --output text)

# SSH into the instance with a command like this...
ssh -i "$KEY_NAME" "ubuntu@$public_dns_name"
# ... then set up the data like this
sudo mount /dev/xvdf /scpc-artifacts
mkdir -p /scpc-artifacts/{data,exp}
git clone https://github.com/sdrobert/scpc.git
cd scpc
git submodule update --init
ln -s $(cd /scpc-artifacts/data; pwd -P)
ln -s $(cd /scpc-artifacts/exp; pwd -P)
source scripts/aws_env.sh
# you can run step-by-step with the -o flag until you've made the
# train_clean_100_train_subset and train_clean_100_test_subset, or you can
# just run until you inevitably fail when you hit the model fitting part b/c
# of the lack of GPUs
./run.sh
# disconnect from the shell and terminate the ec2 instance
aws ec2 terminate-instances --instance-ids $instance_id --output text