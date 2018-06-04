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

bucket_hash=13
bucket="leela-zero-v1"

cd "instance/data/$bucket" || cd "../instance/data/$bucket" || exit 2;
mkdir -p sgf
cd sgf

# Extract all.sgf to "data"
#time sgfsplit -d7 -x 'leela-zero-v1-' ~/Downloads/leela-zero-all.sgf
#ls . | grep '\.sgf$' | xargs grep -o 'PB\[[^]]*\]PW\[[^]]*\]' | tqdm > ../versions
#ls . | grep 'sgf$' | xargs awk '{ if (match($0, /PB\[[^]]*\]PW\[[^]]*\]/)) { print FILENAME ":" substr($0, RSTART, RLENGTH); } nextfile; }' |tqdm > ../versions

hash_to_short_hash='s#(\[[0-9a-f]{8})[0-9a-f]{56}#\1#gi'
leela_strip='s#\[Leela\s*Zero\s*([0-9](\.[0-9]+)*)?\s*(networks)?\s*#[#g'
name_extractor='s#PB\[([^]]*)\]PW\[([^]]*)\]#\1_\2#'
move_notation='s#\([^:]*\):_\?\(.*\)_\?#\1 \2#'
network_dedup='s#\([0-9a-f]\{8\}\)\(_\1\)\?#\1#'

sed ../versions -E -e 's#\[Human\]#[]#' -e "$leela_strip" -e "$hash_to_short_hash" -e "$name_extractor" \
    | sed -n -e "$move_notation" -e "${network_dedup}p" | tqdm > ../raw_moves

# Must be same network
valid_move='^leela-zero-[a-z0-9-]*-[0-9]*\.sgf \([0-9a-f]\{8\}\)$'
grep "$valid_move" ../raw_moves | tqdm > ../moves

# This could be much faster if we used python or something with a hash
#cut ../moves -d' ' -f2 | cut -d'_' -f1 | tqdm | sort -u | xargs -I{} sh -c 'echo "{}"; mkdir -p {}/clean; grep " {}" ../moves | cut -d" " -f1 | tqdm | xargs mv -t {}/clean'
gawk '{ files[$2][$1]; } '\
'END { for (i in files) { '\
'   print "mkdir -p " i "/clean;";'\
'   k = 0; for (j in files[i]) { '\
'       if (k % 1000 == 0) { printf "\nmv -t %s/clean ", i }; '\
'       printf "%s ", j; '\
'       k = k + 1;'\
'   } print;'\
'} }' ../moves | tqdm | xargs -I{} sh -c '{}'


# mkdir -p unknown/clean; ls *.sgf | tqdm | xargs mv -t unknown_net/clean

# For update_models
# cat ../zero-sjeng-org.html | grep -o '<td>[0-9]\{1,3\}</td>.*networks/[a-f0-9]\{64\}.gz' | grep -o '[a-f0-9]\{64\}' | xargs touch -a
