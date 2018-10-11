#!/bin/bash
#
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#
# Tool to sync sgf files from minigo public bucket
# ./rsync-sgf-data.sh -b minigo-pub -r v5-19x19 -m 102 clean data/

function print_help {
    echo "
usage: $0 [-h] [-n] [-t] [-m model] [-b bucket] [-r run] [-c count] type dest

This script rsyncs sgf files from minigo public bucket to local drive

optional arguments:
  -h              Print this help message
  -n              Dry run
  -t              use time based names
  -m model        only sync this models and later (e.g. 102 syncs models >=102)
  -b bucket       Google Cloud bucket, defaults to "minigo-pub"
  -r run          Top level folder in bucket, defaults to "v7-19x19"
  -c count        Sync at most this many files from each directory

Positional arguments
  type  Source directory in bucket {clean, full, both, models}
  dest  Directory to sync files to
"
}

function error {
    echo $1;
    exit 2;
}

BUCKET="minigo-pub"
RUN="v7-19x19"
MIN_MODEL=
DRY_RUN=
MODELS=
TIME_BASED_NAMES=
MAX_COUNT=
while getopts hntm:b:r:c: option
do
 case "${option}" in
   h) print_help; exit 2;;
   n) DRY_RUN=1;;
   t) TIME_BASED_NAMES=1;;
   m) MIN_MODEL=$OPTARG;;
   b) BUCKET=$OPTARG;;
   r) RUN=$OPTARG;;
   c) MAX_COUNT=$OPTARG;;
   *) error "Unexpected option ${option}";;
 esac
done

# Remove parsed arguements
shift $((OPTIND-1))

# Arg checking
if [[ "$#" -ne 2 ]]; then
    print_help;
    exit 2;
fi

TYPE="$1"
DEST="$2"

if [[ ! -d "$DEST" ]]; then
    echo "\"$DEST\" is not a directory";
    print_help;
    exit 2;
fi

EVALS=
CLEAN=1
FULL=1
case "$TYPE" in
  both) ;;
  clean)
    FULL=;;
  full)
    CLEAN=;;
  models)
    MODELS=1;;
  evals)
    EVALS=1;;
  *) error "Unexpected TYPE option $TYPE";;
esac

# gsutil doesn't like // in folder paths which happens if RUN is empty.
base_cloud_path="gs://$(echo "$BUCKET/$RUN/" | sed 's#//#/#g')"
if [[ "$MODELS" ]]; then
    cloud_model_path="${base_cloud_path}models/"
    echo "Syncing model files from $cloud_path, at $(date)"

    dest="$DEST/$RUN/models"
    gsutil -m rsync "$cloud_model_path" "$dest"
    exit 0;
fi

if [[ "$EVALS" ]]; then
    cloud_path="${base_cloud_path}sgf/eval"
    echo "Syncing eval files from $cloud_path, at $(date)"

    # also try syncing a couple of date folders
    TODAY=$(date +"%Y-%m-%d")
    YESTERDAY=$(date --date=yesterday +"%Y-%m-%d")
    TOMORROW=$(date --date=tomorrow +"%Y-%m-%d")

    dest="$DEST/$RUN/eval"
    #mkdir -p "$dest"
    #gsutil -m rsync -r "$cloud_path" "$dest"

    mkdir -p "$dest/$TODAY" "$dest/$YESTERDAY" "$dest/$TOMORROW"
    gsutil -m rsync -r "$cloud_path/$TODAY" "$dest/$TODAY"
    gsutil -m rsync -r "$cloud_path/$YESTERDAY" "$dest/$YESTERDAY"
    gsutil -m rsync -r "$cloud_path/$TOMORROW" "$dest/$TOMORROW"
    exit 0;
fi

# SYNC GAMES
cloud_path="${base_cloud_path}sgf"
partial_dest="$DEST/$RUN/sgf"
echo "Getting models list from $cloud_path"
if [[ "$TIME_BASED_NAMES" ]]; then
    models=$(gsutil ls "$cloud_path/clean" | grep -o "2018-[0-9-]\{8\}/")
else
    models=$(gsutil ls "$cloud_path" | grep -o "00[0-9]\{4\}-[a-z-]*/")
fi

first_model=$(echo "$models" | head -n 1)
last_model=$(echo "$models" | tail -n 1)
echo "Found $(echo "$models" | wc -l) models ($first_model to $last_model)"

if [[ "$MIN_MODEL" ]]; then
    models=$(echo "$models" | sed 's#^\(\([0-9-]*\)\(-.*\|/\)\)$#\2 \1#' |
                    awk -v min="$MIN_MODEL" '$1 >= min { print $2 }')
    echo "Syncing $(echo "$models" | wc -l) models >= $MIN_MODEL"
fi

if [[ -z "$models" ]]; then
    echo "NO MODELS FOUND!"
    exit;
fi

read -p "Do you want to sync \"$cloud_path\" to \"$partial_dest\" [Y]/n: " answer
case $answer in
    [Yy]* ) ;;
    [Nn]* ) exit;;
    * ) echo "Defaulting Yes!";;
esac

function gs_rsync() {
    mkdir -p "$2"
    echo -e "\t\e[1;32m$1 => $2\e[0m"
    echo -e "\t$(date)"

    if [[ "$MAX_COUNT" ]]; then

        existing="$(ls "$2" | wc -l)"
        echo -e "\texisting: $existing"
        if [[ "$existing" -ge "$MAX_COUNT" ]]; then
            return;
        fi
        file_list="$(gsutil ls $1)"
        found="$(echo "$file_list" | wc -l)"
        echo "$(echo "$file_list" | head)"
        echo -e "\tfound: $found"
        if [[ $found -gt 10 ]]; then
            partial_list="$(echo "$file_list" | shuf -n $MAX_COUNT | sort)"
            gsutil -m cp $partial_list "$2"
        fi
    else
        gsutil -m rsync -r "$1" "$2"
    fi
}

echo "$models" | while read model;
do
    echo
    echo -e "\e[1;32mSyncing $model $(date)\e[0m"
    if [[ -z "$DRY_RUN" ]]; then
        if [[ "$TIME_BASED_NAMES" ]]; then
            # NOTE: Wait one hour To ensure uniform sample of TIME_BASED_NAMES
            last_model=$(echo "$models" | tail -n 1)
            if [[ "$MAX_COUNT" ]] && [[ "$last_model" == "$model" ]]; then
                echo -e "\tSkipping recent model: $model"
                continue
            fi

            full_path="full/$model"
            clean_path="clean/$model"
        else
            full_path="$model/full"
            clean_path="$model/clean"
        fi

        if [[ "$CLEAN" ]]; then
            gs_rsync "$cloud_path/$clean_path" "$partial_dest/$clean_path"
        fi
        if [[ "$FULL" ]]; then
            # TODO syncing this before the end of the hour leads to problems :/
            # especially given how cron runs right at the start of the hour.
            gs_rsync "$cloud_path/$full_path" "$partial_dest/$full_path"
        fi
    fi
done
