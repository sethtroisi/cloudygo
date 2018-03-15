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

# Generate model numbers from zero-sjeng.org

#curl http://zero.sjeng.org > zero-sjeng-org.html
#ls import/ | cut -d- -f2 | sed 's#_\(slow\|fast\)##' | sort -u | xargs -I {} sh -c 'num=$(grep -m 1 "{}" zero-sjeng-org.html | sed "s#<tr><td>\([0-9]*\)</td>.*#\1#"); [ ! -z "$num" ] && echo "$(('$bucket_hash' * 1000000 + $num)),{},{},'$bucket',$num,0,0,0,0,0,0"' | sort -n | tee inserts.csv

echo
echo ".mode csv"
echo ".import instance/data/$bucket/inserts.csv models"

# Rename to model numbers
cat inserts.csv | cut -d ',' -f 2,5 | sed -e 's#\(.*\),\(.*\)#rename "s/\1/\2/" import/*.sgf#' | xargs -I {} sh -c '{}'

non_prod="$(ls import/ | grep --color=never '[0-9a-f]\{8\}')"
if [[ ! -z "$non_prod" ]]; then
    echo "$non_prod"
    echo "$(echo "$non_prod" | wc -l) non prod games"
    echo "$non_prod" | xargs -I {} mv import/{} eval_non_prod/
fi
