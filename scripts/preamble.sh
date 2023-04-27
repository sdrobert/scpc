#!/usr/bin/env bash

# common command-line option parser for run.sh and superb_run.sh

source scripts/utils.sh

usage () {
    local ret="${1:-1}"
    echo -e "Usage: $0 [-${HLP_FLG}|${OLY_FLG}|${SRN_FLG}] [-$DAT_FLG DIR]" \
        "[-$EXP_FLG DIR] [-$MDL_FLG MDL] [-$VER_FLG I] [-$PRC_FLG N]"\
        "[-$WRK_FLG N] [-$LIB_FLG DIR] [-$XTR_FLG ARGS]"
    if ((ret == 0)); then
        IFS=","
        cat << EOF 1>&2
$help_message

Options
 -$HLP_FLG                  Display this message and exit
 -$OLY_FLG                  Perform only one step and return
 -$SRN_FLG                  Prefix work-heavy commands with "srun" (i.e.
                            when running nested in an sbatch command)
 -$DAT_FLG DIR              Root data directory (default: $data)
 -$EXP_FLG DIR              Root experiment directory (defalt: $exp)
 -$MDL_FLG {${!MDL2FT[*]}}  
                     Model to train (default: $model)
 -$VER_FLG I                Non-negative integer version number (default: $ver)
 -$PRC_FLG N                Number of processes to spawn in multi-threaded task
                    (default: $p)
 -$WRK_FLG N                Number of workers per process
 -$LIB_FLG DIR              Directory of where librispeech has been
                     downloaded (default: downloads into
                     $data/librispeech/local/data)
 -$XTR_FLG ARGS             Extra args to pass to trainer in fit stage
EOF
    fi
    exit "${1:-1}"
}

# constants
HLP_FLG=h
OLY_FLG=o
SRN_FLG=s
DAT_FLG=d
EXP_FLG=e
MDL_FLG=m
VER_FLG=v
PRC_FLG=p
WRK_FLG=w
LIB_FLG=l
XTR_FLG=x
declare -A FT2ARGS=(
    [raw]="--raw"
    [fbank]=""
    [fbank-80]="--computer-json conf/fbank-80-feats.json"
)
declare -A FT2PAD=(
    [raw]="399"
    [fbank]="0"
    [fbank-80]="0"
)
declare -A FT2DENOM=(
    [raw]="16000"
    [fbank]="100"
    [fbank-80]="100"
)
declare -A FT2SUPERB_ARGS=(
    [fbank]="-g prep/conf/feats/fbank_41.json"
    [fbank-80]="-g conf/fbank-80-feats.json"
)
declare -A MDL2FT
for f in conf/model.*.yaml; do
    exp_name="${f:11:-5}"
    act_name="$(awk '$1 == "name:" {print $2}' "$f")"
    if [ "$exp_name" != "$act_name" ]; then
        echo -e "expected name: in '$f' to be '$exp_name'; got '$act_name';" \
            " ignoring as possible model"
        continue
    fi
    ft="$(awk -v ft="raw" '$1 == "feat_type:" {ft=$2} END {print ft}' "$f")"
    if [ -z "${FT2PAD[$ft]}" ]; then
        echo -e "expected feat_type: in '$f' to be one of ${!FT2PAD[*]}; got" \
            "$ft; ignoring as possible model"
        continue
    fi
    MDL2FT["$exp_name"]="$ft"
done
if [ -z "${MDL2FT["cpc.deft"]}" ]; then
    echo -e "Missing cpc.deft!"
    exit 1
fi

# variables
data="data"
exp="exp"
model="cpc.deft"
libri=
ver=0
nproc=1
nwork=4
only=0
xtra_args=
cmd=
cmd_p=

while getopts "${HLP_FLG}${OLY_FLG}${SRN_FLG}${DAT_FLG}:${EXP_FLG}:${MDL_FLG}:${VER_FLG}:${PRC_FLG}:${WRK_FLG}:${LIB_FLG}:${XTR_FLG}:" opt; do
    case $opt in
        ${HLP_FLG})
            usage 0
            ;;
        ${OLY_FLG})
            only=1
            ;;
        ${SRN_FLG})
            cmd="srun -- "
            cmd_p="srun --ntasks=1 -- "
            ;;
        ${DAT_FLG})
            argcheck_is_writable $opt "$OPTARG"
            data="$OPTARG"
            ;;
        ${EXP_FLG})
            argcheck_is_writable $opt "$OPTARG"
            exp="$OPTARG"
            ;;
        ${MDL_FLG})
            argcheck_is_a_choice $opt "${!MDL2FT[@]}" "$OPTARG"
            model="$OPTARG"
            ;;
        ${VER_FLG})
            argcheck_is_nnint $opt "$OPTARG"
            ver="$OPTARG"
            ;;
        ${PRC_FLG})
            argcheck_is_nat $opt "$OPTARG"
            nproc="$OPTARG"
            ;;
        ${WRK_FLG})
            argcheck_is_nat $opt "$OPTARG"
            nwork="$OPTARG"
            ;;
        ${LIB_FLG})
            argcheck_is_readable $opt "$OPTARG"
            libri="$OPTARG"
            ;;
        ${XTR_FLG})
            xtra_args="$OPTARG"
            ;;
        *)
            usage
            ;;
    esac
done
