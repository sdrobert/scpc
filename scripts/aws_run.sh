#!/bin/bash
# Get instance ID, Instance AZ, Volume ID and Volume AZ 
INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)
INSTANCE_AZ=$(curl -s http://169.254.169.254/latest/meta-data/placement/availability-zone)
AWS_REGION=ca-central-1

VOLUME_ID=$(aws ec2 describe-volumes --region $AWS_REGION --filter "Name=tag:Name,Values=scpc-artifacts" --query "Volumes[].VolumeId" --output text)
VOLUME_AZ=$(aws ec2 describe-volumes --region $AWS_REGION --filter "Name=tag:Name,Values=scpc-artifacts" --query "Volumes[].AvailabilityZone" --output text)

# Proceed if Volume Id is not null or unset
if [ ! -z "$VOLUME_ID" ]; then
		# Check if the Volume AZ and the instance AZ are same or different.
		# If they are different, create a snapshot and then create a new volume in the instance's AZ.
		if [ "$VOLUME_AZ" != "$INSTANCE_AZ" ]; then
				SNAPSHOT_ID=$(aws ec2 create-snapshot \
						--region $AWS_REGION \
						--volume-id $VOLUME_ID \
						--description "`date +"%D %T"`" \
						--tag-specifications 'ResourceType=snapshot,Tags=[{Key=Name,Value=scpc-artifacts-snapshot}]' \
						--query SnapshotId --output text)

				aws ec2 wait --region $AWS_REGION snapshot-completed --snapshot-ids $SNAPSHOT_ID
				aws ec2 --region $AWS_REGION  delete-volume --volume-id $VOLUME_ID

				VOLUME_ID=$(aws ec2 create-volume \
						--region $AWS_REGION \
								--availability-zone $INSTANCE_AZ \
								--snapshot-id $SNAPSHOT_ID \
						--volume-type gp2 \
						--tag-specifications 'ResourceType=volume,Tags=[{Key=Name,Value=scpc-artifacts}]' \
						--query VolumeId --output text)
				aws ec2 wait volume-available --region $AWS_REGION --volume-id $VOLUME_ID
		fi
		# Attach volume to instance
		aws ec2 attach-volume \
			--region $AWS_REGION --volume-id $VOLUME_ID \
			--instance-id $INSTANCE_ID --device /dev/sdf
		sleep 10

        # mount the EBS volume
		mkdir /scpc-artifacts
		mount /dev/xvdf /scpc-artifacts

        # get training code and link into it
		git clone https://github.com/sdrobert/scpc.git
        cd scpc
        git submodule update --init
        ln -s "$(cd /scpc-artifacts/data)"
        ln -s "$(cd /scpc-artifacts/exp)"

        # activate python env
        source scripts/aws_env.sh

        # run a step
        ./run.sh
fi

# After training, clean up by cancelling spot requests and terminating itself
SPOT_FLEET_REQUEST_ID=$(aws ec2 describe-spot-instance-requests --region $AWS_REGION --filter "Name=instance-id,Values='$INSTANCE_ID'" --query "SpotInstanceRequests[].Tags[?Key=='aws:ec2spot:fleet-request-id'].Value[]" --output text)
aws ec2 cancel-spot-fleet-requests --region $AWS_REGION --spot-fleet-request-ids $SPOT_FLEET_REQUEST_ID --terminate-instances
