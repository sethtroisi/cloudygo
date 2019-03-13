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

# find 000* -type f -path '*/clean/*sgf' | sort | tqdm | tar cjf 'clean.tar.bz2' -T -

# Arg checking
if [[ "$#" -ne 1 ]]; then
    echo "Need run arg"
    exit 2;
fi

if [[ "$PWD" = *oneoff* ]]; then
    echo "Run from CloudyGo root"
    exit 2;
fi

run=$1
echo "SGFStrip eval run: \"$run\""

if [[ ! -d "instance/data/$run/eval/" ]]; then
    echo "No eval for $run"
    exit 2;
fi

#if [[ -d "instance/data/$run/eval_clean/" ]]; then
#    echo "eval clean already exists for $run"
#    exit 2;
#fi

cd "instance/data/$run"

echo "Recreate directory structure"
time find -L eval -type d -printf "%P\0" | xargs -0 -I{} mkdir -p eval_clean/{}

echo "Running SGFstrip"
total_sgfs=$(find -L eval -type f -iname '*.sgf' | wc -l)
time find -L eval -type f -iname '*.sgf' -printf "%P\0" \
    | tqdm --delim '\0' --unit "SGF" --total $total_sgfs \
    | xargs -0 -I{} sh -c 'sgfstrip -h C < "eval/{}" > "eval_clean/{}"'

