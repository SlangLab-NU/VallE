#!/usr/bin/env bash

set -eou pipefail

# fix segmentation fault reported in https://github.com/k2-fsa/icefall/issues/674
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

nj=16
stage=-1
stop_stage=3

# We assume dl_dir (download dir) contains the following
# directories and files. If not, they will be downloaded
# by this script automatically.
#
#  - $dl_dir/LJSpeech-1.1

dl_dir=$PWD/download

audio_extractor="Encodec"  # or Fbank
audio_feats_dir=data/tokenized


# . shared/parse_options.sh || exit 1


# All files generated by this script are saved in "data".
# You can safely remove "data" and rerun this script to regenerate it.
mkdir -p data

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

log "dl_dir: $dl_dir"

# if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
#   log "Stage 0: Download data"

#   # If you have pre-downloaded it to /path/to/LJSpeech,
#   # you can create a symlink
#   #
#   ln -sfv /home/data1/VallE/vall-e/egs/uaspeech $dl_dir/UASpeech
#   #
#   # if [ ! -d $dl_dir/LJSpeech-1.1 ];then
#   #   lhotse download UASpeech $dl_dir
#   # fi
# fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  log "Stage 1: Prepare UASpeech manifest"
  # We assume that you have downloaded the UASpeech corpus
  # to $dl_dir/LJSpeech
  mkdir -p data/manifests
  if [ ! -e data/manifests/.uaspeech.done ]; then
    # If lhotse has the prepare_uaspeech in its library try line below
    # lhotse prepare uaspeech $dl_dir/UASpeech data/manifests
    python /home/data1/vall-e.git/VallE/egs/uaspeech/uaspeech.py $dl_dir/UASpeech data/manifests
    touch data/manifests/.uaspeech.done
  fi
fi

# Test/Train split handled in stage 1 preparation
# https://github.com/ffxiong/uaspeech/blob/master/s5_segment/local/prepare_uaspeech_data.sh
# split B1 B3 as training and B2 as test
if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  log "Stage 2: Split UASpeech"

  if [ ! -e data/manifests/uaspeech_recordings_test.jsonl.gz ]; then
    # for manifest in /home/data1/VallE/vall-e/egs/uaspeech/data/manifests/uaspeech_cerebral_recordings_all.jsonl.gz;do
    for manifest in "recordings" "supervisions";do
      echo $manifest
    done
  fi
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  log "Stage 3: ${audio_extractor} UASpeech"

  mkdir -p ${audio_feats_dir}
  if [ ! -e ${audio_feats_dir}/.uaspeech.done ]; then
    python3 bin/tokenizer.py --dataset-parts "uaspeech" --prefix "uaspeech" --suffix "json"\
        --audio-extractor ${audio_extractor} \
        --batch-duration 400 \
        --src-dir "data/manifests" \
        --output-dir "${audio_feats_dir}"
  fi

  # Need to split speaker test set into test/dev sets. Has to be done for each speaker
  # For now it will have hardcoded speakers, for the sake of adapting model

  # total_cuts_test=$(zcat ${audio_feats_dir}/cuts_atypical_speakers_test.jsonl.gz| wc -l)
  # mid_index_test=$((total_cuts_test / 2))
  # echo ${total_cuts_test}
  # echo ${mid_index_test}
  # # dev atypical
  # lhotse subset --last ${mid_index_test}\
  #   ${audio_feats_dir}/cuts_atypical_speakers_test.jsonl.gz \
  #   ${audio_feats_dir}/cuts_atypical_speakers_dev.jsonl.gz
  
  # # test atypical
  # lhotse subset --first ${mid_index_test} \
  #   ${audio_feats_dir}/cuts_atypical_speakers_test.jsonl.gz \
  #   ${audio_feats_dir}/cuts_atypical_speakers_test.jsonl.gz

  touch ${audio_feats_dir}/.uaspeech.done

fi

python3 ./bin/display_manifest_statistics.py --manifest-dir ${audio_feats_dir}