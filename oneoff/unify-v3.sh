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

V3_DIR="../instance/data/v3-9x9/sgf"

if [[ -z "$V3_DIR" || ! -d "$V3_DIR" ]]; then
    echo "$V3_DIR does not exist"
    exit 2;
fi

cd "$V3_DIR"
models=$(ls | grep -o "00[0-9]\{4\}-[a-z-]*")
model_count=$(echo "$models" | wc -l)
first_model=$(echo "$models" | head -n 1)
last_model=$(echo "$models" | tail -n 1)
echo "Found $model_count models ($first_model to $last_model)"

read -p "Do you want to unify v3 to v5 layout? \"$V3_DIR\" y/[N]: " answer
case $answer in
    [Yy]* ) ;;
    [Nn]* ) exit;;
    * ) echo "Defaulting Yes"; #exit;;
esac


count=0
echo "$models" | while read model;
do
    echo "" | tqdm --total $model_count --initial $count --leave False
    count=$((count + 1))

    file_count=$(find "$model" -maxdepth 0 -iname "*.sgf" | wc -l)
    if [[ $file_count -gt 0 ]]; then
        echo "model \"$model\", moving $file_count SGFs"
        mkdir -p "$model/full"
        mv "$model/"*sgf "$model/full/"
    fi

    if [[ -d "$model/full" ]]; then
        mkdir -p "$model/clean"
        cd $model/full
        find . -iname '*.sgf' | tqdm | xargs -I {} sh -c "sgfstrip C < {} > ../clean/{}"
        cd ../..
    fi
done

#time find . -iname '*.sgf' | tqdm | xargs -I {} sh -c "sgfstrip C < {} > ../clean/{}"

