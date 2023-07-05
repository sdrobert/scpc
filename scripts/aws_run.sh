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
    IFS="," echo -e "Usage: $0 [-${HLP_FLG}] [-${CPU_FLG} N] [-${GPU_FLG} NN] [-${MEM_FLG} N]" \
      "[-${CNF_FLG} PTH] [-${RUN_FLG} PTH] [-${URL_FLG} URL] [-- RUN_ARGS]"
    if ((ret == 0)); then
        cat << EOF 1>&2
E.g.: $0 -G 1 -- -m cpc.small -v 1

Compatiblility layer for issuing AWS spot fleet requests. 

Options
 -$HLP_FLG      Display this message and exit
 -$CPU_FLG      Request (at least) this many CPUs in the instance (deft: $ncpu)
 -$GPU_FLG      Request (at least) this many CPUs in the instance (deft: $ngpu)
 -$MEM_FLG      Request (at least) this many MiBs in the instance (deft: $nmib)
 -$CNF_FLG      Path to spot fleet configuration template (deft: $cnf)
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

EOF
    fi
    exit "${1:-1}"
}

# constants
HLP_FLG=h
CPU_FLG=C
GPU_FLG=G
MEM_FLG=M
CNF_FLG=F
RUN_FLG=R
URL_FLG=U

# env vars with defaults
ROLE_NAME="${ROLE_NAME:-scpc-run}"
FLEET_ROLE_NAME="${FLEET_ROLE_NAME:-aws-ec2-spot-fleet-tagging-role}"
POLICY_NAME="${POLICY_NAME:-scpc-run-policy}"
VOLUME_TAG="${VOLUME_TAG:-scpc-artifacts}"
SNAPSHOT_TAG="${SNAPSHOT_TAG:-scpc-snapshots}"

# variables
ncpu=8
ngpu=0
nmib=16000
run_sh="./run.sh"
repo="https://github.com/sdrobert/scpc.git"
cnf=conf/aws-spot-fleet-config.template.json

set -e

while getopts "${HLP_FLG}${CPU_FLG}:${GPU_FLG}:${MEM_FLG}:${CNF_FLG}:${RUN_FLG}:${URL_FLG}:" opt; do
  case $opt in
    ${HLP_FLG})
      usage 0
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
    ${CNF_FLG})
      argcheck_is_readable $opt $OPTARG
      cnf="$OPTARG"
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
    '{
      gsub("<RUN_ARGS>", args);
      gsub("<VOLUME_TAG>", vol_tag);
      gsub("<SNAPSHOT_TAG>", snap_tag);
      gsub("<GIT_REPO>", repo);
      gsub("<RUN_SH>", run_sh);
      print}' \
    scripts/aws_run_internal.template.sh | \
    base64 -w0
  )"

for name in SECURITY_GROUP_ID KEY_NAME AWS_REGION IMAGE_ID AWS_ZONES \
          SNAPSHOT_TAG ROLE_NAME AWS_ACCOUNT_ID FLEET_ROLE_NAME VOLUME_TAG; do
  if [ -z "${!name}" ]; then
    echo "Environment variable '$name' was not set and could not be determined"
    exit 1
  fi
  export "$name"
done
export user_data ncpu ngpu nmib

echo "The remaining arguments '$*' will be passed to $run_sh"
aws ec2 request-spot-fleet --spot-fleet-request-config "$(cat "$cnf" | envsubst)"
