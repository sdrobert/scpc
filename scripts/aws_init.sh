#! /usr/bin/env bash

exit 0
# the below aren't meant to be run exactly as-is. I didn't run them this way.
# However, it should give you a good idea of what to do. Copy and paste what
# you will. Permissions could be tightened.
# Basic idea
#
# Tutorials scavenged:
# - https://aws.amazon.com/blogs/machine-learning/train-deep-learning-models-on-gpus-using-amazon-ec2-spot-instances/

# stores the following constants:
# - KEY_NAME (the name of the ssh key you created via console or whatever)
# - AWS_REGION (the region in which you want everything to run)
# - BUCKET_NAME (OPTIONAL: name of the s3 bucket to store/load results n' stuff from)
[ -f "aws_private/aws_vars.sh" ] && source "$PRIVATE_DIR/aws_vars.sh"

# other, less-sensitive stuff (or can be determined dynamically)
ROLE_NAME=scpc-run
FLEET_ROLE_NAME=aws-ec2-spot-fleet-tagging-role
POLICY_NAME=scpc-run-policy
SNAPSHOT_TAG=scpc-snapshots
EC2_SG_NAME=scpc-ec2-sg
SNAPSHOT_TAG=scpc-artifacts
AWS_ZONES="$(aws ec2 describe-availability-zones --query "AvailabilityZones[?RegionName=='$AWS_REGION'].ZoneName" --output text | tr $'\t' ',')"
DEFT_AWS_ZONE="${AWS_ZONES%%,*}"
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query "Account" --output text)
IMAGE_ID=$(aws ec2 describe-images --region $AWS_REGION --owners amazon --filters 'Name=name,Values=Deep Learning AMI GPU PyTorch 1.13.? (Amazon Linux 2) ????????' 'Name=state,Values=available' --query 'reverse(sort_by(Images, &CreationDate))[:1].ImageId' --output text)
VPC_ID=$(aws ec2 describe-vpcs --filters 'Name=is-default,Values=true' --query 'Vpcs[].VpcId' --output text)
SUBNET_IDS="$(aws ec2 describe-subnets --region $AWS_REGION --filters "Name=vpc-id,Values=${VPC_ID}" "Name=default-for-az,Values=true" --query 'Subnets[].SubnetId' --output text | tr $'\t' ',')"
DEFT_SUBNET_ID="$(aws ec2 describe-subnets --region $AWS_REGION --filters "Name=vpc-id,Values=${VPC_ID}" "Name=default-for-az,Values=true" "Name=availability-zone,Values=$DEFT_AWS_ZONE" --query 'Subnets[].SubnetId' --output text)"
VOL_SIZE=300

# create the EC2 security group
EC2_SG_ID=$(aws ec2 create-security-group \
    --region "$AWS_REGION" \
    --group-name "${EC2_SG_NAME}" \
    --description "Security group for scpc EC2 instance" \
    --vpc-id "$VPC_ID" \
    --output text)
# (can also query id later with command)
# EC2_SG_ID=$(aws ec2 describe-security-groups --group-names $EC2_SG_NAME --query 'SecurityGroups[].GroupId' --output text)

# allow SSH traffic into ec2 sg (feel free to modify this to be more restrictive)
aws ec2 authorize-security-group-ingress \
    --group-id "$EC2_SG_ID" \
    --protocol tcp \
    --port 22 \
    --cidr 0.0.0.0/0 \
    --region "$AWS_REGION"

aws s3api create-bucket \
    --bucket "$BUCKET_NAME" \
    --region "$AWS_REGION" \
    --object-ownership BucketOwnerEnforced \
    --create-bucket-configuration "LocationConstraint=${AWS_REGION}"

# create the volume
aws ec2 create-volume --size $VOL_SIZE --region $AWS_REGION --availability-zone "${AWS_ZONES%%,*}" --volume-type gp2 --tag-specifications "ResourceType=volume,Tags=[{Key=Name,Value=${VOLUME_TAG}}]"

# create the role for the spot instance
aws iam create-role --role-name "$ROLE_NAME" --assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Sid":"","Effect":"Allow","Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]}'
(
    export BUCKET_NAME
    aws iam create-policy \
        --policy-name "$POLICY_NAME"  \
        --policy-document "$(cat "conf/aws-run-policy.json" | envsubst)"
)
aws iam attach-role-policy \
    --policy-arn "arn:aws:iam::$AWS_ACCOUNT_ID:policy/$POLICY_NAME" \
    --role-name "$ROLE_NAME"

# create the role for the spot fleet
aws iam create-role \
     --role-name "$FLEET_ROLE_NAME" \
     --assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Sid":"","Effect":"Allow","Principal":{"Service":"spotfleet.amazonaws.com"},"Action":"sts:AssumeRole"}]}'
aws iam attach-role-policy \
     --policy-arn arn:aws:iam::aws:policy/service-role/AmazonEC2SpotFleetTaggingRole \
     --role-name "$FLEET_ROLE_NAME"
