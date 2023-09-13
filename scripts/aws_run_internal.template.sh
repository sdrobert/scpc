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

BUCKET_NAME='<BUCKET_NAME>'
RUN_ARGS=( <RUN_ARGS> )
IS_DIRTY=<DIRTY>
VOLUME_TAG='<VOLUME_TAG>'
DO_TENSORBOARD=<DO_TENSORBOARD>
GIT_REPO='<GIT_REPO>'
RUN_SH='<RUN_SH>'
SNAPSHOT_TAG='<SNAPSHOT_TAG>'
IS_LEAF=<IS_LEAF>
INSTANCE_ID="$(curl -s http://169.254.169.254/latest/meta-data/instance-id)"
INSTANCE_AZ="$(curl -s http://169.254.169.254/latest/meta-data/placement/availability-zone)"
AWS_REGION="$(curl -s http://169.254.169.254/latest/meta-data/placement/region)"

leaf_volume=

do_cleanup() {
    rv=${1:-$?}
    echo "Finishing with exit code $rv"
    if (($rv == 0)) && [ ! -z "$BUCKET_NAME" ] && [ -d "$em" ]; then
        echo "Finishing up by syncing $em folder"
        aws s3 sync "$em/" "s3://${BUCKET_NAME}/$em/"
    fi
    if $IS_DIRTY; then
        echo "Dirty flag was specified. Not terminating spot fleet request "
        echo "and not stopping the instance"
        exit $rv
    else
        echo "Waiting a bit before cleaning up"
        sleep 120
        echo "Cleaning up spot fleet"
        local request_id="$(aws ec2 describe-spot-instance-requests --region "$AWS_REGION" --filter "Name=instance-id,Values='$INSTANCE_ID'" --query "SpotInstanceRequests[].Tags[?Key=='aws:ec2spot:fleet-request-id'].Value[]" --output text)"
        aws ec2 cancel-spot-fleet-requests \
            --region "$AWS_REGION" \
            --spot-fleet-request-ids "$request_id" \
            --terminate-instances
        exit $rv
    fi
}

# if $IS_LEAF; then
#     echo "Tagging leaf volume"
#     leaf_volume="$(aws ec2 describe-volumes --region $AWS_REGION --filter "Name=attachment.instance-id,Values=$INSTANCE_ID" --filter "Name=attachment.device,Values=/dev/sdf" --query 'Volumes[].Attachments[].VolumeId' --output text || true)"
#     if [ -z "$leaf_volume" ]; then
#         echo "Could not determine leaf volume!"
#         do_cleanup 1
#     fi
#     mods=""
#     $IS_DIRTY && mods="dirty"
#     $IS_LEAF && mods="${mods:+$mods, }leaf"
#     $DO_TENSORBOARD && mods="${mods:+$mods, }tensorboard"
#     name="scpc${mods:+ ($mods)}: ${RUN_SH} ${RUN_ARGS[*]}"
#     aws ec2 create-tags \
#         --region $AWS_REGION \
#         --resources "$leaf_volume" \
#         --tags "Key=Name,Value='$name'" || do_cleanup
# fi

if ! test -e /dev/sdf; then
    if $IS_LEAF; then
        echo "We're a leaf node but /dev/sdf isn't mounted!"
        do_cleanup 1
    fi
    echo "Getting volume IDs and AZ"
    volume_id="$(aws ec2 describe-volumes --region "$AWS_REGION" --filter "Name=tag:Name,Values=$VOLUME_TAG" --query "Volumes[].VolumeId" --output text)"
    volume_az="$(aws ec2 describe-volumes --region "$AWS_REGION" --filter "Name=tag:Name,Values=$VOLUME_TAG" --query "Volumes[].AvailabilityZone" --output text)"

    if [ -z "$volume_id" ]; then
        echo "Missing volume ID!"
        do_cleanup 1
    fi
    if [ -z "$volume_az" ]; then
        echo "Missing volume AZ!"
        do_cleanup 1
    fi

    if [ "$volume_az" != "$INSTANCE_AZ" ]; then
        echo "Mismatch between volume ($volume_az) and instance ($INSTANCE_AZ) AZ!"
        echo "Creating snapshot"
        SNAPSHOT_ID="$(aws ec2 create-snapshot \
            --region "$AWS_REGION" \
            --volume-id "$volume_id" \
            --description "`date +"%D %T"`" \
            --tag-specifications "ResourceType=snapshot,Tags=[{Key=Name,Value=$SNAPSHOT_TAG}]" \
            --query SnapshotId \
            --output text)"
        if [ -z "$SNAPSHOT_ID" ]; then
            echo "Snapshot not created"
            do_cleanup 1
        fi
        echo "Waiting for snapshot to complete"
        aws ec2 wait snapshot-completed --region "$AWS_REGION" --snapshot-ids "$SNAPSHOT_ID" || do_cleanup
        echo "Waiting for volume to become available"
        aws ec2 wait volume-available --region "$AWS_REGION" --volume-ids "$volume_id" || do_cleanup
        echo "Deleting old volume"
        aws ec2 --region "$AWS_REGION"  delete-volume --volume-id "$volume_id" || do_cleanup
        echo "Creating new volume"
        volume_id="$(aws ec2 create-volume \
            --region "$AWS_REGION" \
            --availability-zone "$INSTANCE_AZ" \
            --snapshot-id "$SNAPSHOT_ID" \
            --volume-type gp2 \
            --tag-specifications "ResourceType=volume,Tags=[{Key=Name,Value=$VOLUME_TAG}]" \
            --query VolumeId \
            --output text)"
        if [ -z "$volume_id" ]; then
            echo "Volume not created"
            do_cleanup 1
        fi
        volume_az="$INSTANCE_AZ"
    fi

    echo "Waiting for volume to become available"
    aws ec2 wait volume-available \
        --region "$AWS_REGION" \
        --volume-ids "$volume_id" || do_cleanup

    echo "Attaching volume"
    aws ec2 attach-volume \
        --region "$AWS_REGION" --volume-id "$volume_id" \
        --instance-id "$INSTANCE_ID" --device /dev/sdf || do_cleanup
    aws ec2 wait volume-in-use \
        --region "$AWS_REGION" --volume-id "$volume_id" || do_cleanup
    sleep 1
fi

file_s="$(sudo file -sL /dev/sdf)"
if [[ "$file_s" =~ 'filesystem' ]]; then
    echo "Volume already a filesystem"
else
    echo "Formatting volume"
    sudo mkfs -t xfs /dev/sdf || do_cleanup
fi



mkdir -p /scpc-ebs || do_cleanup

if ! mount | grep -q /dev/sdf; then
    echo "Mounting EBS volume"
    mount /dev/sdf /scpc-ebs || do_cleanup
    mkdir -p /scpc-ebs/{data,exp}
fi

mkdir -p /scpc
cd /scpc

if ! git status 2> /dev/null; then
    echo "Cloning training source"
    git clone --depth 1 "$GIT_REPO" . || do_cleanup
fi

git submodule update --init || do_cleanup

ln -sf "$(cd /scpc-ebs/exp; pwd -P)"
ln -sf "$(cd /scpc-ebs/data; pwd -P)"

echo "Activating and updating python environment"
source activate pytorch
if [ ! -f .install_complete ]; then
    $DO_TENSORBOARD && conda install tensorboard
    conda install -c sdrobert pydrobert-kaldi pydrobert-param
    pip install "git+https://github.com/sdrobert/pydrobert-pytorch.git" "git+https://github.com/sdrobert/pydrobert-speech"
    pip install '.[all]'
    touch .install_complete
fi

if [ -z "$BUCKET_NAME" ]; then
    echo "Bucket unavailable for exp/"
else
    echo "Bucket specified! Will sync down exp"
    echo "Determining model directory..."
    em="$(source scripts/preamble.sh "${RUN_ARGS[@]}" > /dev/null 2> /dev/null; echo "$em")"
    if [ -z "$em" ]; then
        echo "Model directory cannot be inferred. Will not sync!"
    else
        if [ ! -f ".sync_complete" ]; then
            echo "Model directory inferred as $em. Will sync from bucket"
            rm -rf "$em"
            mkdir -p "$em"
            aws s3 sync "s3://${BUCKET_NAME}/$em/" "$em/" || do_cleanup
            touch .sync_complete
        else
            echo "Looks like we already synced this..."
        fi
    fi
fi

dl="$data/librispeech"
dlf="$data/librispeech/$ft"

if $DO_TENSORBOARD; then
    echo "Starting tensorboard in the background"
    mkdir -p exp/tb_logs
    tensorboard --logdir=exp/tb_logs &
fi

echo "Running with args ${RUN_ARGS[*]}"
$RUN_SH -zx "--quiet" "${RUN_ARGS[@]}"
do_cleanup
