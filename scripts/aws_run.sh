#!/usr/bin/env bash

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

if ! aws help > /dev/null; then
    echo "No CLI installed! Cannot do anything!"
    exit 1
fi

[ -f "aws_private/aws_vars.sh" ] && source "aws_private/aws_vars.sh"
source scripts/utils.sh

usage () {
    local ret="${1:-1}"
    IFS="," echo -e "Usage: $0 [-${HLP_FLG}] [-${LEF_FLG}] [-${DTY_FLG}]" \
      "[-${CPU_FLG} N] [-${GPU_FLG} NN] [-${MEM_FLG} N]" \
      "[-${RUN_FLG} PTH] [-${URL_FLG} URL] [-- RUN_ARGS]"
    if ((ret == 0)); then
        cat << EOF 1>&2
E.g.: $0 -G 1 -- -m cpc.small -v 1

Compatiblility layer for issuing AWS spot fleet requests.

Options
 -$HLP_FLG      Display this message and exit
 -$LEF_FLG      Run as a leaf (see below)
 -$DTY_FLG      If set, will not terminate the spot fleet request nor the
         instance after running.
 -$CPU_FLG      Request (at least) this many CPUs in the instance (deft: $ncpu)
 -$GPU_FLG      Request (at least) this many CPUs in the instance (deft: $ngpu)
 -$MEM_FLG      Request (at least) this many MiBs in the instance (deft: $nmib)
 -$RUN_FLG      Which script to run in repository root (deft: $run_sh)
 -$URL_FLG      URL to git repo to clone (deft: $repo)

In addition, the following environment variables must be set before calling.
Parentheses indicate their value, with XXX indicating a value has been set
but is potentially sensitive.

  AWS_ACCOUNT_ID ${AWS_ACCOUNT_ID:+(XXX)}
  AWS_REGION ${AWS_REGION:+(${AWS_REGION})}
  AWS_ZONES ${AWS_ZONES:+(${AWS_ZONES})}
  FLEET_ROLE_NAME ${FLEET_ROLE_NAME:+(${FLEET_ROLE_NAME})}
  IMAGE_ID ${IMAGE_ID:+(${IMAGE_ID})}
  KEY_NAME ${KEY_NAME:+(${KEY_NAME})}
  POLICY_NAME ${POLICY_NAME:+(${POLICY_NAME})}
  ROLE_NAME ${ROLE_NAME:+($ROLE_NAME)}
  SECURITY_GROUP_ID ${SECURITY_GROUP_ID:+(XXX)}
  SNAPSHOT_TAG ${SNAPSHOT_TAG:+(${SNAPSHOT_TAG})}
  VOLUME_TAG ${VOLUME_TAG:+(${VOLUME_TAG})}

ROLE_NAME, FLEET_ROLE_NAME, POLICY_NAME, VOLUME_TAG, and SNAPSHOT_TAG should
all have default values. AWS_ZONES, AWS_ACCOUNT_ID, and IMAGE_ID may be
inferred later with aws commands (assuming appropriate privileges).

Leaf mode:

By default, the spot fleet will attach to the "ground truth" EBS volume,
replicating it to the instance's availability and destroying the previous
version when necessary. This ensures only one version of artifacts will be
valid at a given time. However, this method cannot be parallelized. By
enabling leaf mode with the flag -${LEF_FLG}, ground truth will be copied to
the instance's local EBS volume without mounting it. The local EBS volume
will not be destroyed whe the spot fleet request is terminated. While the
new volume will be available on AWS, it is no longer part of the run pipeline
-- hence, a "leaf." It is recommended that all data preparation steps be
completed on the ground truth EBS so that no work is duplicated.

If the environment variable SNAPSHOT_ID has been set ${SNAPSHOT_ID:+(yes)}, this snapshot id
will be copied into the local volume instead of creating a new one.

EOF
    fi
    exit "${1:-1}"
}

# constants
HLP_FLG=h
LEF_FLG=L
DTY_FLG=D
CPU_FLG=C
GPU_FLG=G
MEM_FLG=M
RUN_FLG=R
URL_FLG=U
DEFT_EBS_VOL_SIZE=32  # The volume size used for booting when the ground-true
DEFT_CONF_TEMPLATE=conf/aws-spot-fleet-config.template.json
LEAF_CONF_TEMPLATE=conf/aws-spot-fleet-config-leaf.template.json

# env vars with defaults
ROLE_NAME="${ROLE_NAME:-scpc-run}"
FLEET_ROLE_NAME="${FLEET_ROLE_NAME:-aws-ec2-spot-fleet-tagging-role}"
POLICY_NAME="${POLICY_NAME:-scpc-run-policy}"
VOLUME_TAG="${VOLUME_TAG:-scpc-artifacts}"
SNAPSHOT_TAG="${SNAPSHOT_TAG:-scpc-snapshots}"

# variables
ncpu=4
ngpu=0
nmib=16000
run_sh="./run.sh"
repo="https://github.com/sdrobert/scpc.git"
is_leaf=false
is_dirty=false

set -e

while getopts "${HLP_FLG}${LEF_FLG}${DTY_FLG}${CPU_FLG}:${GPU_FLG}:${MEM_FLG}:${RUN_FLG}:${URL_FLG}:" opt; do
  case $opt in
    ${HLP_FLG})
      usage 0
      ;;
    ${LEF_FLG})
      is_leaf=true
      ;;
    ${DTY_FLG})
      is_dirty=true
      ;;
    ${CPU_FLG})
      argcheck_is_nat $opt $OPTARG
      ncpu="$OPTARG"
      ;;
    ${GPU_FLG})
      argcheck_is_nnint $opt $OPTARG
      ngpu="$OPTARG"
      ;;
    ${MEM_FLG})
      argcheck_is_nat $opt $OPTARG
      nmib="$OPTARG"
      ;;
    ${RUN_FLG})
      run_sh="$OPTARG"
      ;;
    ${URL_FLG})
      repo="$OPTARG"
      ;;
    *)
      usage
      ;;
  esac
done

shift $((OPTIND-1))

if [[ "$*" =~ "'" ]]; then
    echo "The remaining arguments $* cannot contain the ' character!"
    return 1
fi

if [ -z "$AWS_ZONES" ]; then
  echo "AWS_ZONES empty. Trying to infer AWS_ZONES from 'aws ec2 describe-availability-zones'"
  AWS_ZONES="$(aws ec2 describe-availability-zones --query "AvailabilityZones[?RegionName=='$AWS_REGION'].ZoneName" --output text | tr $'\t' ',' || true)"
  if [ -z "$AWS_ZONES" ]; then
    echo "Failed to infer AWS_ZONES!"
    exit 1
  fi
fi
if [ -z "$AWS_ACCOUNT_ID" ]; then
  echo "AWS_ACCOUNT_ID empty. Trying to infer AWS_ACCOUNT_ID from 'aws sts get-caller-identity'"
  AWS_ACCOUNT_ID="$(aws sts get-caller-identity --query "Account" --output text || true)"
  if [ -z "$AWS_ACCOUNT_ID" ]; then
    echo "Failed to infer AWS_ACCOUNT_ID!"
    exit 1
  fi
fi
if [ -z "$IMAGE_ID" ]; then
  echo "IMAGE_ID empty. Trying to infer IMAGE_ID from 'aws ec2 describe-images'"
  IMAGE_ID="$(aws ec2 describe-images --region $AWS_REGION --owners amazon --filters 'Name=name,Values=Deep Learning AMI GPU PyTorch 1.13.? (Amazon Linux 2) ????????' 'Name=state,Values=available' --query 'reverse(sort_by(Images, &CreationDate))[:1].ImageId' --output text || true)"
  if [ -z "$IMAGE_ID" ]; then
    echo "Failed to infer IMAGE_ID!"
    exit 1
  fi
fi


user_data="$(
  awk -v args="$(printf " '%s' " "$@")" \
    -v vol_tag="${VOLUME_TAG}" \
    -v snap_tag="${SNAPSHOT_TAG}" \
    -v repo="${repo}" \
    -v run_sh="${run_sh}" \
    -v dirty="$dirty" \
    '{
      gsub("<RUN_ARGS>", args);
      gsub("<VOLUME_TAG>", vol_tag);
      gsub("<SNAPSHOT_TAG>", snap_tag);
      gsub("<GIT_REPO>", repo);
      gsub("<RUN_SH>", run_sh);
      gsub("<DIRTY>", dirty);
      print}' \
    scripts/aws_run_internal.template.sh | \
    base64 -w0
  )"

echo "The remaining arguments '$*' will be passed to $run_sh"

if [ $ngpu != 0 ]; then
  acc_man='"nvidia"'
else
  acc_man=
fi

if $is_leaf; then
  echo "Will make a leaf"
  cnf="${LEAF_CONF_TEMPLATE}"
  if [ -z "$SNAPSHOT_ID" ]; then
    echo "SNAPSHOT_ID is not set. Creating a snapshot. First, determine volume"
    volume_id="$(aws ec2 describe-volumes --region "$AWS_REGION" --filter "Name=tag:Name,Values=${VOLUME_TAG}" --query "Volumes[].VolumeId" --output text || true)"
    if [ -z "$volume_id" ]; then
      echo "Could not determine volume"
      exit 1
    fi
    echo "Creating snapshot"
    SNAPSHOT_ID="$(aws ec2 create-snapshot --region "$AWS_REGION" --volume-id "$volume_id" --description "`date +"%D %T"`" --tag-specifications "ResourceType=snapshot,Tags=[{Key=Name,Value=${SNAPSHOT_TAG}}]" --query SnapshotId --output text || true)"
    if [ -z "$SNAPSHOT_ID" ]; then
      echo "Could not create snapshot"
      exit 1
    fi
  fi
  echo "Waiting for snapshot to complete"
  aws ec2 wait snapshot-completed --region "$AWS_REGION" --snapshot-ids "$SNAPSHOT_ID"
else
  echo "Will attach ground truth to the instance"
  cnf="${DEFT_CONF_TEMPLATE}"
fi

# export
for name in SECURITY_GROUP_ID KEY_NAME AWS_REGION IMAGE_ID AWS_ZONES \
          SNAPSHOT_TAG ROLE_NAME AWS_ACCOUNT_ID FLEET_ROLE_NAME VOLUME_TAG; do
  if [ -z "${!name}" ]; then
    echo "Environment variable '$name' was not set and could not be determined"
    exit 1
  fi
  export "$name"
done
export user_data ncpu ngpu nmib SNAPSHOT_ID ebs_volume_size delete_ebs acc_man

echo "Requesting spot fleet instance"
aws ec2 request-spot-fleet --spot-fleet-request-config "$(cat "$cnf" | envsubst)"
