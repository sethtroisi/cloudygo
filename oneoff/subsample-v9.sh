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


#find sgf/full/ -type f | tqdm | xargs grep 'C\[Resign Threshold: ' > resign_thresholds
RUN="v9-19x19"
RUN_DIR="../instance/data/$RUN"
RESIGN_THRESHOLDS="resign_thresholds"

PERCENT=10

if [[ -z "$RUN_DIR" || ! -d "$RUN_DIR" ]]; then
    echo "$RUN_DIR does not exist"
    exit 2;
fi

if [[ ! -f "$RUN_DIR/$RESIGN_THRESHOLDS" ]]; then
    echo "didn't find '$RESIGN_THRESHOLDS' file"
    exit 2;
fi

cd "$RUN_DIR"
hours=$(ls sgf/full | grep -o "201[89]-[0-9][0-9]-[0-9][0-9]-[0-9][0-9]")
hour_count=$(echo "$hours" | wc -l)
first_hour=$(echo "$hours" | head -n 1)
last_hour=$(echo "$hours" | tail -n 1)
echo "Found $hour_count hours ($first_hour to $last_hour)"

read -p "Do you want to subsample $RUN to 10%? \"$RUN_DIR\" y/[N]: " answer
case $answer in
    [Yy]* ) ;;
    [Nn]* ) exit;;
    * ) echo "Defaulting Yes"; exit;;
esac

count=0
echo "$hours" | while read hour;
do
    count=$((count + 1))

    all_files=$(find "sgf/full/$hour" -maxdepth 1 -iname "*.sgf")
    file_count=$(echo "$all_files" | wc -l)
    resign_files=$(grep "sgf/full/$hour" "$RESIGN_THRESHOLDS")
    resign_count=$(echo "$resign_files" | wc -l)

#   TODO only do if > 1000 files
#    eligable=$all_files
#    eligable_count=$(echo "$eligable" | wc -l)
#    new_count=$(echo "(0.9 * $eligable_count)/1" | bc)
#    echo "$hour contains $file_count files, $resign_count resign records, sampling $eligable_count, rm $new_count"
#    echo "$eligable" | shuf | head -n "$new_count" | xargs rm

#    if [[ $file_count -gt 0 && $file_count -eq $resign_count ]]; then
#        eligable=$(echo "$resign_files" | grep -v ': -1' | grep -o '^[^:]*')
#        eligable_count=$(echo "$eligable" | wc -l)
#        new_count=$(echo "(0.9 * $eligable_count)/1" | bc)
#        echo "$hour contains $file_count files, $resign_count resign records, sampling $eligable_count, rm $new_count"
#        echo "$eligable" | shuf | head -n "$new_count" | xargs rm
#    fi
done
