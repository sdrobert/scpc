#! /usr/bin/env bash

exit 0
# the below aren't meant to be run exactly as-is. I didn't run them this way.
# However, it should give you a good idea of what to do. Copy and paste what
# you will
#
# Follows loosely along with
# https://aws.amazon.com/blogs/machine-learning/train-deep-learning-models-on-gpus-using-amazon-ec2-spot-instances/

# stores the following constants:
# - SECURITY_GROUP_ID
# - KEY_NAME
# - AWS_REGION
[ -f "aws_private/aws_vars.sh" ] && source "$PRIVATE_DIR/aws_vars.sh"

# other, less-sensitive stuff
ROLE_NAME=scpc-run
FLEET_ROLE_NAME=aws-ec2-spot-fleet-tagging-role
POLICY_NAME=scpc-run-policy
VOLUME_TAG=scpc-artifacts
SNAPSHOT_TAG=scpc-snapshots
VOL_SIZE=200
AWS_ZONES="$(aws ec2 describe-availability-zones --query "AvailabilityZones[?RegionName=='$AWS_REGION'].ZoneName" --output text | tr $'\t' ',')"
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query "Account" --output text)
IMAGE_ID=$(aws ec2 describe-images --region $AWS_REGION --owners amazon --filters 'Name=name,Values=Deep Learning AMI GPU PyTorch 1.13.? (Amazon Linux 2) ????????' 'Name=state,Values=available' --query 'reverse(sort_by(Images, &CreationDate))[:1].ImageId' --output text)

# this script injects variable values into template files, printing to stdout
build_private_version() {
    local inf="$1"
    shift
    if [ ! -f "$inf" ]; then
        echo -e "'$inf' is not a file"
        return 1
    fi
    if [[ "$*" =~ "'" ]]; then
        echo -e "Arguments $* cannot contain ' character"
        return 1
    fi
    local user_data="$(awk -v args="$(printf " '%s' " "$@")" -v vol_tag="${VOLUME_TAG}" -v snap_tag="${SNAPSHOT_TAG}" '{gsub("<RUN_ARGS>", args); gsub("<VOLUME_TAG>", vol_tag); gsub("<SNAPSHOT_TAG>", snap_tag); print}' scripts/aws_run.sh | base64 -w0)"
    {
        export SECURITY_GROUP_ID KEY_NAME AWS_REGION VOL_SIZE IMAGE_ID
        export user_data ROLE_NAME AWS_ACCOUNT_ID FLEET_ROLE_NAME VOLUME_TAG
        export GPU_LAUNCH_TEMPLATE_NAME CPU_LAUNCH_TEMPLATE_NAME AWS_ZONES
        export SNAPSHOT_TAG
        cat "$inf" | envsubst
    }
}

# create the volume
aws ec2 create-volume --size $VOL_SIZE --region $AWS_REGION --availability-zone "${AWS_ZONES%%,*}" --volume-type gp2 --tag-specifications "ResourceType=volume,Tags=[{Key=Name,Value=${VOLUME_TAG}}]"

# create the relevant roles
aws iam create-role --role-name "$ROLE_NAME" --assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Sid":"","Effect":"Allow","Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]}'
aws iam create-policy \
    --policy-name "$POLICY_NAME"  \
    --policy-document "$(build_private_version "conf/aws-run-policy.json")"
aws iam attach-role-policy \
    --policy-arn "arn:aws:iam::$AWS_ACCOUNT_ID:policy/$POLICY_NAME" \
    --role-name "$ROLE_NAME"
aws iam create-role \
     --role-name "$FLEET_ROLE_NAME" \
     --assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Sid":"","Effect":"Allow","Principal":{"Service":"spotfleet.amazonaws.com"},"Action":"sts:AssumeRole"}]}'
aws iam attach-role-policy \
     --policy-arn arn:aws:iam::aws:policy/service-role/AmazonEC2SpotFleetTaggingRole \
     --role-name "$FLEET_ROLE_NAME"

# run a cpu step
aws ec2 request-spot-fleet --spot-fleet-request-config "$(build_private_version conf/aws-cpu-spot-fleet-config.json -o)"

# run a 4-gpu step
aws ec2 request-spot-fleet --spot-fleet-request-config "$(build_private_version conf/aws-4gpu-spot-fleet-config.json -o)"