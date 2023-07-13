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
    IFS="," echo -e "Usage: $0 [-${HLP_FLG}] [-${DTY_FLG}]" \
      "[-${PNT_FLG}] [-${CPU_FLG} N] [-${GPU_FLG} NN] [-${MEM_FLG} N]" \
      "[-${RUN_FLG} PTH] [-${URL_FLG} URL] [-- RUN_ARGS]"
    if ((ret == 0)); then
        cat << EOF 1>&2
E.g.: $0 -G 1 -- -m cpc.small -v 1

Compatiblility layer for issuing AWS spot fleet requests.

Options
 -$HLP_FLG      Display this message and exit
 -$LEF_FLG      Run as a leaf (see below)
 -$PNT_FLG      Print the spot fleet request to stdout and exit (WARNING: will
         display potentially sensitive information)
 -$DTY_FLG      If set, will not terminate the spot fleet request nor the
         instance after running.
 -$TSB_FLG      If set, start a tensorboard server on port 6006
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
  EC2_SG_ID ${EC2_SG_ID:+(XXX)}
  SNAPSHOT_TAG ${SNAPSHOT_TAG:+(${SNAPSHOT_TAG})}
  VOLUME_TAG ${VOLUME_TAG:+(${VOLUME_TAG})}

ROLE_NAME, FLEET_ROLE_NAME, POLICY_NAME, SNAPSHOT_TAG, and SNAPSHOT_TAG and
VOLUME_TAG should all have default values. AWS_ZONES, AWS_ACCOUNT_ID,
EC2_SG_ID, SUBNET_IDS, and IMAGE_ID may be inferred later with aws commands
(assuming appropriate privileges).

EOF
    fi
    exit "${1:-1}"
}

# constants
CPU_FLG=C
DTY_FLG=D
GPU_FLG=G
HLP_FLG=h
LEF_FLG=L
MEM_FLG=M
PNT_FLG=P
RUN_FLG=R
TSB_FLG=T
URL_FLG=U

# env vars with defaults
ROLE_NAME="${ROLE_NAME:-scpc-run}"
FLEET_ROLE_NAME="${FLEET_ROLE_NAME:-aws-ec2-spot-fleet-tagging-role}"
POLICY_NAME="${POLICY_NAME:-scpc-run-policy}"
EC2_SG_NAME="${EC2_SG_NAME:-scpc-ec2-sg}"
VOLUME_TAG="${VOLUME_TAG:-scpc-artifacts}"
SNAPSHOT_TAG="${SNAPSHOT_TAG:-scpc-snapshots}"

# variables
ncpu=4
ngpu=0
nmib=16000
run_sh="./run.sh"
repo="https://github.com/sdrobert/scpc.git"
is_dirty=false
is_print=false
is_leaf=false
do_tensorboard=false
cnf=conf/aws-spot-fleet-config.template.json

set -e

while getopts "${HLP_FLG}${PNT_FLG}${LEF_FLG}${DTY_FLG}${TSB_FLG}${CPU_FLG}:${GPU_FLG}:${MEM_FLG}:${RUN_FLG}:${URL_FLG}:" opt; do
  case $opt in
    ${HLP_FLG})
      usage 0
      ;;
    ${PNT_FLG})
      is_print=true
      ;;
    ${LEF_FLG})
      is_leaf=true
      ;;
    ${DTY_FLG})
      is_dirty=true
      ;;
    ${TSB_FLG})
      do_tensorboard=true
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
if [ -z "$EC2_SG_ID" ]; then
  echo "EC2_SG_ID is empty. Trying to infer EC2_SG_ID from 'aws ec2 describe-security-groups'"
  EC2_SG_ID=$(aws ec2 describe-security-groups --group-names $EC2_SG_NAME --query 'SecurityGroups[].GroupId' --output text)
  if [ -z "$EC2_SG_ID" ]; then
    echo "Failed to infer EC2_SG_ID!"
    exit 1
  fi
fi
if [ -z "$SUBNET_IDS" ]; then
  echo "SUBNET_IDS is empty. Trying to infer SUBNET_IDS from default vpc and 'aws ec2 describe-subnets'"
  VPC_ID="${VPC_ID:-$(aws ec2 describe-vpcs --filters 'Name=is-default,Values=true' --query 'Vpcs[].VpcId' --output text)}"
  SUBNET_IDS="$(aws ec2 describe-subnets --region $AWS_REGION --filters "Name=vpc-id,Values=${VPC_ID}" "Name=default-for-az,Values=true" --query 'Subnets[].SubnetId' --output text | tr $'\t' ',')"
  if [ -z "$SUBNET_IDS" ]; then
    echo "Failed to infer SUBNET_IDS!"
    exit 1
  fi
fi
if $is_leaf; then
  echo "Will make a leaf"
  cnf="conf/aws-spot-fleet-config-leaf.template.json"
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
fi



user_data_raw="$(
  awk -v args="$(printf " '%s' " "$@")" \
    -v dirty="${is_dirty}" \
    -v volume="${VOLUME_TAG}" \
    -v tb="${do_tensorboard}" \
    -v repo="${repo}" \
    -v run_sh="${run_sh}" \
    -v snapshot="${SNAPSHOT_TAG}" \
    -v leaf="${is_leaf}" \
    '{
      gsub("<RUN_ARGS>", args);
      gsub("<DIRTY>", dirty);
      gsub("<VOLUME_TAG>", volume);
      gsub("<DO_TENSORBOARD>", tb);
      gsub("<GIT_REPO>", repo);
      gsub("<RUN_SH>", run_sh);
      gsub("<SNAPSHOT_TAG>", snapshot);
      gsub("<IS_LEAF>", leaf);
      print}' \
    scripts/aws_run_internal.template.sh
  )"
user_data="$(echo "$user_data_raw" | base64 -w0)"

echo "The remaining arguments '$*' will be passed to $run_sh"

if [ $ngpu != 0 ]; then
  acc_man='"nvidia"'
else
  acc_man=
fi

# export
for name in SUBNET_IDS AWS_ACCOUNT_ID FLEET_ROLE_NAME EC2_SG_ID ROLE_NAME KEY_NAME IMAGE_ID AWS_ZONES; do
  if [ -z "${!name}" ]; then
    echo "Environment variable '$name' was not set and could not be determined"
    exit 1
  fi
  export "$name"
done
export SNAPSHOT_ID user_data ncpu ngpu nmib acc_man
mods=""
$is_dirty && mods="dirty"
$is_leaf && mods="${mods:+$mods, }leaf"
$do_tensorboard && mods="${mods:+$mods, }tensorboard"
export name="scpc${mods:+ ($mods)}: ${run_sh} $*"


request_config="$(cat "$cnf" | envsubst)"

if $is_print; then
  echo "-------------------------------------------"
  echo "The request config:"
  echo "-------------------------------------------"
  echo "$request_config" 
  echo
  echo "-------------------------------------------"
  echo "The user data:"
  echo "-------------------------------------------"
  echo "$user_data_raw"
else
  echo "Requesting spot fleet instance"
  aws ec2 request-spot-fleet --spot-fleet-request-config "$request_config"
fi
