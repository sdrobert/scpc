set -e

source scripts/utils.sh

usage () {
    local ret="${1:-1}"
    echo -e "Usage: $0 [-${HLP_FLG}|${OLY_FLG}] [-$DAT_FLG DIR] [-$EXP_FLG DIR] [-$MDL_FLG MDL] [-$VER_FLG I] [-$PRC_FLG N] [-$LIB_FLG DIR]"
    if ((ret == 0)); then
        IFS=','
        cat << EOF 1>&2
Options
 -$HLP_FLG                  Display this message and exit
 -$OLY_FLG                  Perform only one step and return
 -$DAT_FLG DIR              Root data directory (default: $data)
 -$EXP_FLG DIR              Root experiment directory (defalt: $exp)
 -$MDL_FLG {${!MDL2FT[*]}}      Model to train (default: $model)
 -$VER_FLG I                Non-negative integer version number (default: $ver)
 -$PRC_FLG N                Number of threads to spawn in bash pipes
                    (default: $p)
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
DAT_FLG=d
EXP_FLG=e
MDL_FLG=m
VER_FLG=v
PRC_FLG=p
LIB_FLG=l
XTR_FLG=x
declare -A MDL2FT=(
    [fbank-p12]="fbank"
    [cpc.deft]="raw"
    [cpc.small]="raw"
    [cpc.trans]="raw"
)
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

# variables
data="data"
exp="exp"
model="cpc.deft"
libri=
ver=0
nproc=1
only=0
xtra_args=

while getopts "${HLP_FLG}${OLY_FLG}${DAT_FLG}:${EXP_FLG}:${MDL_FLG}:${VER_FLG}:${PRC_FLG}:${LIB_FLG}:${XTR_FLG}:" opt; do
    case $opt in
        ${HLP_FLG})
            usage 0
            ;;
        ${OLY_FLG})
            only=1
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

# download librispeech if necessary
if [ -z "$libri" ] &&  [ ! -f "$dl/.complete" ]; then
    # n.b. don't need to know where librispeech is if already done init_word
    libri="$dl/local/data"
    if [ ! -f "$libri/.complete" ]; then
        # Download librispeech
        python prep/librispeech.py "$dl" download
        touch "$libri/.complete"
        ((only)) && exit 0
    fi
fi

# download abxLS-dataset
if [ ! -f "$dz/.complete" ]; then
    # FIXME(sdrobert): this is entirely redundant. the files are the same
    # as the librispeech dev/test partitions, just fewer of them and of WAV
    # format.
    zrc datasets:pull abxLS-dataset
    touch "$dz/.complete"
    ((only)) && exit 0
fi

# initial prep of dataset
if [ ! -f "$dl/.complete" ]; then
    python prep/librispeech.py data/librispeech preamble \
        --speakers-are-readers --exclude-subsets "$libri"
    python prep/librispeech.py data/librispeech init_word "$libri"
    touch "$dl/.complete"
    ((only)) && exit 0
fi

# compute features and store in feat/ subdir
if [ ! -f "$dlf/.complete" ]; then
    python prep/librispeech.py \
        data/librispeech torch_dir wrd $ft ${FT2ARGS[$ft]} --compute-up-to=100
    touch "$dlf/.complete"
    ((only)) && exit 0
fi

# put phone alignments into ali/ subdir
if [ ! -f "$dlf/train_clean_100/ali/.complete" ]; then
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
        unzip -cq resources/converted_aligned_phones.zip | \
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

# fix alignments and get info file
if [ ! -f "$dlf/ext/train_clean_100.info.ark" ]; then
    failed=0
    get-torch-spect-data-dir-info --fix ${FT2PAD[$ft]} \
        "$dlf/"{train_clean_100,ext/train_clean_100.info.ark} || failed=1
    if ((falied)); then
        rm -f "$dlf/ext/train_clean_100.info.ark"
        exit 1
    fi
    ((only)) && exit 0
fi

# pull out the subsets of train_clean_100 used for training and testing
for x in train test; do
    if [ ! -f "$dlf/train_clean_100_${x}_subset/.complete" ]; then
        subset-torch-spect-data-dir \
            "$dlf/train_clean_100"{,_${x}_subset} \
            --utt-list-file resources/train_clean_100_${x}_subset.txt
        touch "$dlf/train_clean_100_${x}_subset/.complete"
        ((only)) && exit 0
    fi
done

# train model
if [ ! -f "$em/best.pt" ]; then
    scpc \
        --read-model-yaml "conf/model.$model.yaml" \
        fit \
            "$dlf/train_clean_100_train_subset" \
            "$dlf/train_clean_100_test_subset" \
            @conf/trainer.args.txt \
            "--default_root_dir=$exp" \
            "--version=$ver" $xtra_args
    [ -f "$em/best.pt" ] || exit 1
    ((only)) && exit 0
fi

# compute predictions
for x in dev_clean dev_other test_clean test_other; do
    if [ ! -f "$pdl/$x/.complete" ]; then
        mkdir -p "$pdl/$x"
        scpc \
            --read-model-yaml "conf/model.$model.yaml" \
            predict --numpy --device=cuda \
            "$em/best.pt" "$dlf/$x" "$pdl/$x"
        touch "$pdl/$x/.complete"
        ((only)) && exit 0
    fi
done

# set up zerospeech abxLS submission
if [ ! -f "$zs/.complete" ]; then
    rm -rf "$zs"
    zrc submission:init abxLS "$zs"
    scpc --read-model-yaml "conf/model.$model.yaml" info | \
        awk -v denom=${FT2DENOM[$ft]} '
BEGIN {spf=0}
NR == FNR && $1 == "downsampling_factor" {spf=$2 / denom}
NR != FNR {if ($1 == "feature_size:") $2=spf; print}' \
            - conf/params.template.yaml > "$zs/params.yaml"
    awk '
BEGIN {sd="\"my great model\""}
NR == FNR && $1 == "system_description:" {$1=""; split($0, x, "#"); sd=x[1]}
NR != FNR {if ($1 == "system_description:") $2=sd; print}
' conf/{model.$model.yaml,meta.template.yaml} > "$zs/meta.yaml"
    awk -v "di=$pdl/" -v "do_=$zs/" '{$1="\""di$1"\""; $2="\""do_$2"\""; print}' \
        resources/libri_to_abxLS.map | xargs -P $nproc -I{} sh -c 'cp -f {}'
    touch "$zs/.complete"
    find "$pdl" -name '*.npy' -delete  # we don't need these anymore
    ((only)) && exit 0
fi

# score zerospeech abxLS submission
if [ ! -f "$zs/scores/.complete" ]; then
    zrc benchmarks:run abxLS "$zs"
    touch "$zs/scores/.complete"
    ((only)) && exit 0
fi


# # check the mean and stddev of phoneme lengths, excluding silence
# print-torch-ali-data-dir-length-moments \
#     data/librispeech/raw/train_clean_100/ali --std --exclude-ids 0
# # returns mean=1362.625 std=806.739
# # sum of three phoneme Gaussians:
# #   mean=3*1362.625=4087.875 std=sqrt(3)*806.739=1397.313

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