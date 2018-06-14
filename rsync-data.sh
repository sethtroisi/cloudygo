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
usage: $0 [-h] [-n] [-m model] [-b bucket] [-r run] type dest

This script rsyncs sgf files from minigo public bucket to local drive

optional arguments:
  -h              Print this help message
  -n              Dry run
  -b bucket       Google Cloud bucket, defaults to "minigo-pub"
  -r run          Top level folder in bucket, defaults to "v7-19x19"
  -m model        only sync this models and later (e.g. 102 syncs models >=102)

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
while getopts hnm:b:r: option
do
 case "${option}" in
   h) print_help; exit 2;;
   n) DRY_RUN=1;;
   b) BUCKET=$OPTARG;;
   r) RUN=$OPTARG;;
   m) MIN_MODEL=$OPTARG;;
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

exclude=''
exclude_re=''
EVALS=
case "$TYPE" in
  both) ;;
  clean)
    exclude='-x'
    exclude_re='.*\/?full\/';;
  full)
    exclude='-x'
    exclude_re='.*\/?clean\/';;
  models)
    MODELS="1";;
  evals)
    EVALS="1";;
  *) error "Unexpected TYPE option $TYPE";;
esac

# gsutil doesn't like // in folder paths which happens if RUN is empty.
base_cloud_path="gs://$(echo "$BUCKET/$RUN/" | sed 's#//#/#')"
if [[ ! -z $MODELS ]]; then
    cloud_path="$base_cloud_path""models/"
    echo "Syncing model files from $cloud_path, at $(date)"

    dest=$(readlink -f "$DEST/$RUN/models")
    gsutil rsync "$cloud_path" "$dest"
    exit 0;
fi

if [[ ! -z $EVALS ]]; then
    cloud_path="$base_cloud_path""sgf/eval"
    echo "Syncing eval files from $cloud_path, at $(date)"

    dest=$(readlink -f "$DEST/$RUN/eval")
    gsutil -m rsync -r "$cloud_path" "$dest"
    exit 0;
fi

echo "excluding: \"$exclude_re\""

cloud_path="$base_cloud_path""sgf/"
echo "Getting models list from $cloud_path"
models=$(gsutil ls "$cloud_path" | grep -o "00[0-9]\{4\}-[a-z-]*/")
first_model=$(echo "$models" | head -n 1)
last_model=$(echo "$models" | tail -n 1)
echo "Found $(echo "$models" | wc -l) models ($first_model to $last_model)"

if [[ ! -z $MIN_MODEL ]]; then
    models=$(echo "$models" | sed 's#^\(\([0-9]*\)-.*\)$#\2 \1#' |
                    awk -v min="$MIN_MODEL" '$1 >= min { print $2 }')
    echo "Syncing $(echo "$models" | wc -l) models >= $MIN_MODEL"
fi

read -p "Do you want to sync \"$cloud_path\" to \"$DEST\" [Y]/n: " answer
case $answer in
    [Yy]* ) ;;
    [Nn]* ) exit;;
    * ) echo "Defaulting Yes!";;
esac

partial_dest=$(readlink -f "$DEST/$RUN/sgf")

echo "$models" | while read model;
do
    echo "syncing $model $exclude $exclude_re, $(date)"
    if [[ -z "$DRY_RUN" ]]; then
        mkdir -p "$partial_dest/$model"
        gsutil -m rsync $exclude $exclude_re -r "$cloud_path$model" "$partial_dest/$model"
    fi
done
