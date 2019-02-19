#!/usr/bin/env bash
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

set -euo pipefail

bucket="leela-zero"

cd "instance/data/$bucket"

PER_FOLDER=5000

extract_sgf_to_folder() {
    # eval, sgf, ...
    folder_name="$1"
    combined="$2"
    echo "Extracting \"$combined\" to \"$folder_name\""

    mkdir -p "$folder_name"
    cd "$folder_name"

#    files_list="$(ls)"
#    num_files=$(echo "$files_list" | wc -l)
    num_files=12180000
    if [ "$num_files" -le 1 ]; then
        echo "Splitting \"$combined\" ($(du -h "$combined" | cut -f1))"
        /usr/bin/time -f "Splitting took %e seconds" \
            sgfsplit -d8 -x "${bucket}-" "$combined"
        files_list="$(ls)"
        num_files=$(echo "$files_list" | wc -l)
    fi

    if [ "$num_files" -ge 20000 ]; then
#        echo "Found $num_files files"
        tmp_file_list="../${folder_name}_files_tmp"
#        echo "$files_list" > "$tmp_file_list"

        # assumes files_list is sorted and no directories exist yet
        first=$(head -n 1 $tmp_file_list | grep -o '[1-9][0-9]*')
        last=$(tail  -n 1 $tmp_file_list | grep -o '[0-9]*')

        # TODO do something if first not a multiple of $PER_FOLDER

        # Create folder per X thousand
        first_folder=$(($first/$PER_FOLDER*$PER_FOLDER))
        echo "$first to $last, starting with $first_folder"

        for lower in $(seq $first_folder $PER_FOLDER $last); do
            echo "$lower" # so tqdm can track progress
            mkdir -p $lower
            set +e
            seq -f "${bucket}-%08.0f.sgf" $lower $(($lower + $PER_FOLDER - 1)) | xargs --no-run-if-empty mv -t $lower
            set -e
        done | tqdm --desc "Seperating" --total $(($num_files / $PER_FOLDER)) | wc
#        echo "Done seperating"
    fi
    cd ..
}

save_player_info() {
    # eval, sgf, ...
    folder_name="$1"

    # Create a list of files to player names
    player_file="${folder_name}_file_to_player"
    if [ ! -f "$player_file" ]; then
        echo "Saving player names to \"$player_file\""
        # This sort is only pseudo good"
        find "$folder_name" -type f -name '*.sgf' \
            | sort \
            | xargs grep -H -m1 -o 'PB\[[^]]*\]PW\[[^]]*\]' \
            | tqdm --desc "PB/PW lookup" > "$player_file"
    fi

    players() {
        cat "$player_file" | cut -d':' -f2-
    }
    group() {
        sort | uniq -c | sort -n
    }

    player_combos=$(players | group | wc -l)
    unique_player_combos=$(players | sed 's/Leela\s*Zero\s*[0-9]\+\(\.[0-9]\+\)*\s\+//g' | group | wc -l)
    echo "Player pairs: $player_combos, $unique_player_combos unique"
    echo
}

# Extract all_match.sgf to "data"
ALL_MATCH_PATH="/media/eights/big-ssd/rsync/all_match.sgf"
ALL_SGF_PATH="/media/eights/big-ssd/rsync/all_fixed.sgf"

#echo "Eval"
#extract_sgf_to_folder "eval" "$ALL_MATCH_PATH"
#save_player_info      "eval"

echo "Self Play"
extract_sgf_to_folder "sgf" "$ALL_SGF_PATH"
save_player_info      "sgf"
