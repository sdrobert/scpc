set -e

source scripts/utils.sh

usage () {
    local ret="${1:-1}"
    echo -e "Usage: $0 [-${HLP_FLG}|${OLY_FLG}|${SRN_FLG}] [-$DAT_FLG DIR]" \
        "[-$EXP_FLG DIR] [-$MDL_FLG MDL] [-$VER_FLG I] [-$PRC_FLG N]"\
        "[-$WRK_FLG N] [-$LIB_FLG DIR] [-$XTR_FLG ARGS]"
    if ((ret == 0)); then
        IFS=","
        cat << EOF 1>&2
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
)
declare -A FT2PAD=(
    [raw]="399"
    [fbank]="0"
)
declare -A FT2DENOM=(
    [raw]="16000"
    [fbank]="100"
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

ft="${MDL2FT[$model]}"
dl="$data/librispeech"
dlf="$data/librispeech/$ft"
dz="$data/zerospeech"
em="$exp/$model/version_$ver"
pdl="$em/predict/librispeech"
zs="$em/zrc/librispeech"

# for zerospeech-benchmarks pkg
export APP_DIR="$dz/local/data"
export TEMP_DIR="$TMPDIR"

if [ -z "$libri" ] &&  [ ! -f "$dl/.complete" ]; then
    libri="$dl/local/data"
    if [ ! -f "$libri/.complete" ]; then
        echo "Downloading librispeech"
        $cmd_p python prep/librispeech.py "$dl" download \
            --files \
                {test,dev}-{clean,other}.tar.gz \
                train-clean-100.tar.gz \
                librispeech-vocab.txt
        touch "$libri/.complete"
        ((only)) && exit 0
    fi
fi

if [ ! -f "$dz/.complete" ]; then
    # FIXME(sdrobert): this is entirely redundant. the files are the same
    # as the librispeech dev/test partitions, just fewer of them and of WAV
    # format.
    echo "Downloading zerospeech abxLS"
    $cmd_p zrc datasets:pull abxLS-dataset
    touch "$dz/.complete"
    ((only)) && exit 0
fi

if [ ! -f "$dl/.complete" ]; then
    echo "Performing common prep of librispeech"
    $cmd_p python prep/librispeech.py "$dl" preamble \
        --speakers-are-readers --exclude-subsets "$libri"
    $cmd_p python prep/librispeech.py "$dl" init_word "$libri"
    touch "$dl/.complete"
    ((only)) && exit 0
fi

if [ ! -f "$dlf/.complete" ]; then
    echo "Computing $ft features of librispeech"
    $cmd_p python prep/librispeech.py \
        "$dl" torch_dir wrd $ft ${FT2ARGS[$ft]} --compute-up-to=100
    touch "$dlf/.complete"
    ((only)) && exit 0
fi

if [ ! -f "$dlf/train_clean_100/ali/.complete" ]; then
    echo "Moving per-frame alignments to data dir for $ft"
    rm -f "$dlf/train_clean_100/ali/.complete-"*
    if [ "$ft" = "raw" ]; then
        # convert 10ms frame alignments to sample alignments and store the
        # results in train_clean_100/ali
        #
        # N.B. --snip-edges=true for kaldi, which means frames = (samps -
        #   400) // 160 + 1, or samps <= 160 * (frames - 1) + 400
        #
        # since the first frame starts at sample 0, the extra 240 to 399
        # samples are at the end of the recording. We thus use the alignment of
        # the final frame for an additional 399 frames, then crop using
        # get-torch-spect-data-dir-info.
        align2frames() {
            awk -v spf=160 -v pad=${FT2PAD[$ft]} -v i=$i -v I=$nproc '
(NR + i - 2) % I == 0 {
printf "lbi-%s", $1;
for (n=2; n <= NF; ++n) for (m=0; m < spf; ++m) printf " %s", $n;
for (m=0; m < pad; ++m) printf " %s", $NF;
printf "\n";
}'
        }
    else
        align2frames() {
            awk -v i=$i -v I=$nproc '(NR + i - 2) % I == 0 {print;}'
        }
    fi
    for i in $(seq 1 $nproc); do
        $cmd_p unzip -cq resources/converted_aligned_phones.zip | \
            align2frames | \
            write-table-to-torch-dir \
                -i iv -o long \
                'ark,t,s,o,cs:-' \
                "$dlf/train_clean_100/ali" && \
            touch "$dlf/train_clean_100/ali/.complete-$i" &
    done
    wait
    for i in $(seq 1 $nproc); do
        if [ ! -f "$dlf/train_clean_100/ali/.complete-$i" ]; then
            echo -e "Process $i/$I failed!"
            rm -f "$dlf/train_clean_100/ali/.complete-"*
            exit 1
        fi
    done
    rm -f "$dlf/train_clean_100/ali/.complete-"*
    touch "$dlf/train_clean_100/ali/.complete"
    ((only)) && exit 0
fi

if [ ! -f "$dlf/ext/train_clean_100.info.ark" ]; then
    echo "Fixing alignments and getting info file"
    failed=0
    $cmd_p get-torch-spect-data-dir-info --fix ${FT2PAD[$ft]} \
        "$dlf/"{train_clean_100,ext/train_clean_100.info.ark} || failed=1
    if ((falied)); then
        rm -f "$dlf/ext/train_clean_100.info.ark"
        exit 1
    fi
    ((only)) && exit 0
fi

for x in train test; do
    if [ ! -f "$dlf/train_clean_100_${x}_subset/.complete" ]; then
        echo "Making $x subset of train_clean_100"
        rm -rf "$dlf/train_clean_100_${x}_subset"
        $cmd_p subset-torch-spect-data-dir --num-workers=$nwork \
            "$dlf/train_clean_100"{,_${x}_subset} \
            --utt-list-file resources/train_clean_100_${x}_subset.txt
        touch "$dlf/train_clean_100_${x}_subset/.complete"
        ((only)) && exit 0
    fi
done

if [ ! -f "$em/model.yaml" ]; then
    # this first step ensures conf/model.$model.yaml can be read as-is by
    # the scpc command. The second joins all the default configuration values
    # with the modified ones. The latter ensures the model always trains with
    # a specific configuration, even if the defaults are changed later.
    echo "Checking configuration conf/model.$model.yaml parses"
    $cmd_p scpc fit --read-model-yaml "conf/model.$model.yaml" -h > /dev/null
    echo "Writing $model configuration to $em/model.yaml"
    mkdir -p "$em"
    $cmd_p scpc fit --print-model-yaml | \
        combine-yaml-files --quiet --nested \
            - "conf/model.$model.yaml" "$em/model.yaml"
    ((only)) && exit 0
fi

if [ ! -f "$em/best.ckpt" ]; then
    echo "Training $model model"
    $cmd scpc \
        fit \
            --read-model-yaml "$em/model.yaml" \
            "$dlf/train_clean_100_train_subset" \
            "$dlf/train_clean_100_test_subset" \
            --root-dir "$exp" \
            "--version=$ver" "--num-workers=$nwork" $xtra_args
    [ -f "$em/best.ckpt" ] || exit 1
    echo "Deleting intermediate checkpoints of $model"
    find "$em/" -name '*.ckpt' -not -name 'best.ckpt' -delete
    ((only)) && exit 0
fi

for x in dev_clean dev_other test_clean test_other; do
    if [ ! -f "$pdl/$x/.complete" ]; then
        echo "Computing predictions for $x parition using $model model"
        mkdir -p "$pdl/$x"
        $cmd_p scpc \
            predict --numpy \
            "$em/best.ckpt" "$dlf/$x" "$pdl/$x"
        touch "$pdl/$x/.complete"
        ((only)) && exit 0
    fi
done

if [ ! -f "$zs/.complete" ]; then
    echo "Constructing abxLS zerospeech submission using $model model"
    rm -rf "$zs"
    $cmd_p zrc submission:init abxLS "$zs"
    $cmd_p scpc info "$em/best.ckpt" | \
        awk -v denom=${FT2DENOM[$ft]} '
BEGIN {spf=0}
NR == FNR && $1 == "downsampling_factor" {spf=$2 / denom}
NR != FNR {if ($1 == "feature_size:") $2=spf; print}' \
            - conf/params.template.yaml > "$zs/params.yaml"
    $cmd_p awk '
BEGIN {sd="\"my great model\""}
NR == FNR && $1 == "system_description:" {$1=""; split($0, x, "#"); sd=x[1]}
NR != FNR {if ($1 == "system_description:") $2=sd; print}
' "$em/model.yaml" conf/meta.template.yaml > "$zs/meta.yaml"
    awk -v "di=$pdl/" -v "do_=$zs/" '{$1="\""di$1"\""; $2="\""do_$2"\""; print}' \
        resources/libri_to_abxLS.map | xargs -P $nwork -I{} sh -c 'cp -f {}'
    touch "$zs/.complete"
    find "$pdl" -name '*.npy' -delete  # we don't need these anymore
    ((only)) && exit 0
fi

if [ ! -f "$zs/scores/.complete" ]; then
    echo "Scoring abxLS zerospeech submissing using $model model"
    $cmd_p zrc benchmarks:run abxLS "$zs"
    touch "$zs/scores/.complete"
    ((only)) && exit 0
fi


# # check the average number of 10ms frames in an utterance
# unzip -cq resources/converted_aligned_phones.zip | \
#     awk 'BEGIN {n = 0; c = 0} {n+=NF - 1; c+=1} END {print n / c}'
# # returns 1267.12
# # 160 frames/sec: 7.9195 secs/utt
# # 16,000 samps/sec: 126,712 samps/utt
# # 20,840 samps/win: 6.187109375 win/utt
# # to get 8 win per batch, you need 2 utt to be safe
# # to get 16 win per batch, you need 3 utt to be safe
# # to get 32 win per batch, you need 6 utt to be safe


# # check the mean and stddev of phoneme lengths, excluding silence
# print-torch-ali-data-dir-length-moments \
#     data/librispeech/raw/train_clean_100/ali --std --exclude-ids 0
# # returns mean=1362.625 std=806.739
# #  - nearest multiple of 160: 1280 (9 frames)
# # sum of three phoneme Gaussians:
# #   mean=3*1362.625=4087.875 std=sqrt(3)*806.739=1397.313
# # 1 phone 95% confidence interval = 2,976.103
# #  - nearest multiple of 160: 3,040 (19 frames)
# # 3 phone 95% confidence interval = 6,882.501
# #  - nearest multiple of 160: 6,880 (43 frames)
# # 64 win/batch * (20,480 samps/win / 3,040 samps/win) ~ 432 win (108 / gpu)
# # 64 win/batch * (20,480 samps/win / 6,880 samps/win) ~ 192 win (48 / gpu)


# # the receptive field of a coefficient w/ K conv layers = r_1
# # r_K = kernel_k
# # r_k = kernel_k + stride_k (r_{k+1} - 1)
# # kernel_5=4,  stride_5=2, r_5 = 4
# # kernel_4=4,  stride_4=2, r_4 = 4 + 2 * 3 = 10
# # kernel_3=4,  stride_3=2, r_3 = 4 + 2 * 9 = 22
# # kernel_2=8,  stride_2=4, r_2 = 8 + 4 * 21 = 92
# # kernel_1=10, stride_1=5, r_1 = 10 + 5 * 91 = 465 ~ 3 frames
