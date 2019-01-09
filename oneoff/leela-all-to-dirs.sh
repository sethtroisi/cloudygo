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

bucket="leela-zero-v1"

cd "instance/data/$bucket" || cd "../instance/data/$bucket" || exit 2;
mkdir -p eval models

# For update_models
# curl http://zero.sjeng.org/ > zero-sjeng-org.html
# cd models
# cat ../zero-sjeng-org.html | grep -o '<td>[0-9]\{1,3\}</td>.*networks/[a-f0-9]\{64\}.gz' | grep -o '[a-f0-9]\{64\}' | xargs touch -a
# cat zero-sjeng-org.html | grep -o '<td>[0-9]\{1,3\}</td>.*networks/[a-f0-9]\{64\}.gz' | sed -n 's#^<td>\([0-9]\{1,3\}\)</td><td>\(20[0-9 :-]*\)</td>.*/networks/\([0-9a-f]\{64\}.gz\)#\1\t\2\t\3#p' > names.txt
# cd ..


# Extract all.sgf to "data"
# cd eval
# time sgfsplit -d7 -x 'leela-zero-v1-' ~/Downloads/leela-zero-all.sgf
# ls . | grep '\.sgf$' | xargs grep -m1 -o 'PB\[[^]]*\]PW\[[^]]*\]' | tqdm > ../versions
# cd ..

# python leela-eval-process.py
# wc raw_moves nonprod_moves

# python leela-model-importer.py
