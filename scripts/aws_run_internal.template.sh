#!/bin/bash

# Copyright 2023 Sean Robertson
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# WARNING! This script is called on the start up of a spot-fleet instance.
# Do not try to call it directly

echo "Beginning run in $(pwd -P)"

if ! aws help > /dev/null; then
    echo "No CLI!"
fi

RUN_ARGS=( <RUN_ARGS> )
IS_DIRTY=<DIRTY>
EFS_NAME=<EFS_NAME>

INSTANCE_ID="$(curl -s http://169.254.169.254/latest/meta-data/instance-id)"
INSTANCE_AZ="$(curl -s http://169.254.169.254/latest/meta-data/placement/availability-zone)"
AWS_REGION="$(curl -s http://169.254.169.254/latest/meta-data/placement/region)"
EFS_ID=$(aws efs describe-file-systems --region $AWS_REGION --query "FileSystems[?not_null(Tags[?Value == \`$EFS_NAME\`])].FileSystemId" --output text)


do_cleanup() {
    if $IS_DIRTY; then
        echo "Dirty flag was specified. Not terminating spot fleet request "
        echo "and not stopping the instance"
        exit
    else
        echo "Waiting a bit before cleaning up"
        sleep 120
        echo "Cleaning up spot fleet"
        local request_id="$(aws ec2 describe-spot-instance-requests --region "$AWS_REGION" --filter "Name=instance-id,Values='$INSTANCE_ID'" --query "SpotInstanceRequests[].Tags[?Key=='aws:ec2spot:fleet-request-id'].Value[]" --output text)"
        aws ec2 cancel-spot-fleet-requests --region "$AWS_REGION" --spot-fleet-request-ids $request_id --terminate-instances
        exit
    fi
}

# if ! test -e /dev/sdf; then
#     echo "Getting volume IDs and AZ"
#     volume_id="$(aws ec2 describe-volumes --region "$AWS_REGION" --filter "Name=tag:Name,Values=<VOLUME_TAG>" --query "Volumes[].VolumeId" --output text)"
#     volume_az="$(aws ec2 describe-volumes --region "$AWS_REGION" --filter "Name=tag:Name,Values=<VOLUME_TAG>" --query "Volumes[].AvailabilityZone" --output text)"

#     if [ -z "$volume_id" ]; then
#         echo "Missing volume ID!"
#         do_cleanup
#     fi
#     if [ -z "$volume_az" ]; then
#         echo "Missing volume AZ!"
#         do_cleanup
#     fi

#     if [ "$volume_az" != "$INSTANCE_AZ" ]; then
#         echo "Mismatch between volume ($volume_az) and instance ($INSTANCE_AZ) AZ!"
#         echo "Creating snapshot"
#         SNAPSHOT_ID="$(aws ec2 create-snapshot --region "$AWS_REGION" --volume-id "$volume_id" --description "`date +"%D %T"`" --tag-specifications 'ResourceType=snapshot,Tags=[{Key=Name,Value=<SNAPSHOT_TAG>}]' --query SnapshotId --output text)"
#         if [ -z "$SNAPSHOT_ID" ]; then
#             echo "Snapshot not created"
#             do_cleanup
#         fi
#         echo "Waiting for snapshot to complete"
#         aws ec2 wait snapshot-completed --region "$AWS_REGION" --snapshot-ids "$SNAPSHOT_ID" || do_cleanup
#         echo "Waiting for volume to become available"
#         aws ec2 wait volume-available --region "$AWS_REGION" --volume-ids "$volume_id" || do_cleanup
#         echo "Deleting old volume"
#         aws ec2 --region "$AWS_REGION"  delete-volume --volume-id "$volume_id" || do_cleanup
#         echo "Creating new volume"
#         volume_id="$(aws ec2 create-volume --region "$AWS_REGION" --availability-zone "$INSTANCE_AZ" --snapshot-id "$SNAPSHOT_ID" --volume-type gp2 --tag-specifications 'ResourceType=volume,Tags=[{Key=Name,Value=<VOLUME_TAG>}]' --query VolumeId --output text)"
#         if [ -z "$volume_id" ]; then
#             echo "Volume not created"
#             do_cleanup
#         fi
#         volume_az="$INSTANCE_AZ"
#     fi

#     echo "Waiting for volume to become available"
#     aws ec2 wait volume-available --region "$AWS_REGION" --volume-ids "$volume_id" || do_cleanup

#     echo "Attaching volume"
#     aws ec2 attach-volume \
#         --region "$AWS_REGION" --volume-id "$volume_id" \
#         --instance-id "$INSTANCE_ID" --device /dev/sdf || do_cleanup
#     aws ec2 wait volume-in-use \
#         --region "$AWS_REGION" --volume-id "$volume_id" || do_cleanup
#     sleep 1
# fi

# file_s="$(sudo file -sL /dev/sdf)"
# if [[ "$file_s" =~ 'filesystem' ]]; then
#     echo "Volume already a filesystem"
# else
#     echo "Formatting volume"
#     sudo mkfs -t xfs /dev/sdf || do_cleanup
# fi

mkdir -p /scpc-artifacts || do_cleanup

if ! mount | grep -q /scpc-artifacts; then
    echo "Installing amazon efs utils"
    sudo yum install amazon-efs-utils -y || do_cleanup
    echo "Mounting EFS volume"
    mount -t efs -o tls $EFS_ID /scpc-artifacts || do_cleanup
fi

echo "Cloning training source"
git clone <GIT_REPO>
cd scpc
git submodule update --init

mkdir -p /scpc-artifacts/{data,exp}
ln -s "$(cd /scpc-artifacts/data; pwd -P)"
ln -s "$(cd /scpc-artifacts/exp; pwd -P)"

echo "Activating and updating python environment"
source activate pytorch
conda install tensorboard
conda install -c coml virtual-dataset zerospeech-benchmarks zerospeech-libriabx2 zerospeech-tde
conda install -c sdrobert pydrobert-kaldi pydrobert-param
pip install "git+https://github.com/sdrobert/pydrobert-pytorch.git@scpc" "git+https://github.com/sdrobert/pydrobert-speech"
pip install '.[all]'

# run tensorboard as a background server
echo "Starting tensorboard in the background"
mkdir -p exp/tb_logs
tensorboard --logdir=exp/tb_logs &

echo "Running with args ${RUN_ARGS[*]}"
<RUN_SH> "${RUN_ARGS[@]}" -x "--quiet"
do_cleanup
