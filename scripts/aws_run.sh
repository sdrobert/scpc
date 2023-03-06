#!/bin/bash

echo "Beginning run in $(pwd -P)"

if ! aws help > /dev/null; then
    echo "No CLI!"
fi

RUN_ARGS=( <RUN_ARGS> )

INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)
INSTANCE_AZ=$(curl -s http://169.254.169.254/latest/meta-data/placement/availability-zone)
AWS_REGION=$(curl -s http://169.254.169.254/latest/meta-data/placement/region)

echo "Getting volume IDs and AZ"
volume_id=$(aws ec2 describe-volumes --region $AWS_REGION --filter "Name=tag:Name,Values=scpc-artifacts" --query "Volumes[].VolumeId" --output text)
VOLUME_AZ=$(aws ec2 describe-volumes --region $AWS_REGION --filter "Name=tag:Name,Values=scpc-artifacts" --query "Volumes[].AvailabilityZone" --output text)

if [ -z "$volume_id" ]; then
    echo "Missing volume ID"
fi
if [ -z "$VOLUME_AZ" ]; then
    echo "Missing volume AZ"
fi

# Proceed if Volume Id is not null or unset
if [ ! -z "$volume_id" ] && [ ! -z "$VOLUME_AZ" ]; then
        # Check if the Volume AZ and the instance AZ are same or different.
        # If they are different, create a snapshot and then create a new volume in the instance's AZ.
        if [ "$VOLUME_AZ" != "$INSTANCE_AZ" ]; then
                echo "Mismatch between volume and instance AZ! Moving!"
                SNAPSHOT_ID=$(aws ec2 create-snapshot --region "$AWS_REGION" --volume-id "$volume_id" --description "`date +"%D %T"`" --tag-specifications 'ResourceType=snapshot,Tags=[{Key=Name,Value=scpc-artifacts-snapshot}]' --query SnapshotId --output text)
                if [ ! -z "$SNAPSHOT_ID" ]; then
                    aws ec2 wait --region "$AWS_REGION" snapshot-completed --snapshot-ids "$SNAPSHOT_ID" || echo "Failed to wait for snapshot to start up"
                    aws ec2 --region "$AWS_REGION"  delete-volume --volume-id "$volume_id" || echo "Failed to delete old volume"

                    volume_id=$(aws ec2 create-volume \
                            --region "$AWS_REGION" \
                                    --availability-zone "$INSTANCE_AZ" \
                                    --snapshot-id "$SNAPSHOT_ID" \
                            --volume-type gp2 \
                            --tag-specifications 'ResourceType=volume,Tags=[{Key=Name,Value=scpc-artifacts}]' \
                            --query VolumeId --output text)
                    if [ ! -z "$volume_id" ]; then
                        aws ec2 wait volume-available --region "$AWS_REGION" --volume-id "$volume_id" || echo "Failed to wait for volume availability"
                    else
                        echo "Could not create volume from snapshot!"
                    fi
                else
                    echo "Could not create snapshot!"
                fi
        fi

        echo "Attaching volume to this instance"
        aws ec2 attach-volume \
            --region "$AWS_REGION" --volume-id "$volume_id" \
            --instance-id "$INSTANCE_ID" --device /dev/sdf || echo "Could not attach volume to instance"
        sleep 10

        file_s="$(sudo file -sL /dev/sdf)"
        if [[ "$file_s" =~ 'filesystem' ]]; then
            echo "Already a filesystem"
        else
            echo "Formatting"
            sudo mkfs -t xfs /dev/sdf
        fi

        echo "Mounting EBS volume"
        mkdir /scpc-artifacts
        mount /dev/sdf /scpc-artifacts

        echo "Cloning training source"
        git clone https://github.com/sdrobert/scpc.git
        cd scpc
        git submodule update --init
        mkdir -p /scpc-artifacts/{data,exp}
        ln -s "$(cd /scpc-artifacts/data; pwd -P)"
        ln -s "$(cd /scpc-artifacts/exp; pwd -P)"

        echo "Activating python environment"
        source scripts/aws_env.sh

        # run tensorboard as a background server
        echo "Starting tensorboard in the background"
        mkdir -p exp/tb_logs
        tensorboard --logdir=exp/tb_logs &
        sleep 10

        echo "Running with args ${RUN_ARGS[*]}"
        ./run.sh "${RUN_ARGS[@]}"
fi

echo "Cleaning up spot fleet"
SPOT_FLEET_REQUEST_ID=$(aws ec2 describe-spot-instance-requests --region $AWS_REGION --filter "Name=instance-id,Values='$INSTANCE_ID'" --query "SpotInstanceRequests[].Tags[?Key=='aws:ec2spot:fleet-request-id'].Value[]" --output text)
aws ec2 cancel-spot-fleet-requests --region $AWS_REGION --spot-fleet-request-ids $SPOT_FLEET_REQUEST_ID --terminate-instances
