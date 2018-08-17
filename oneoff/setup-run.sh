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
echo "Setting up run: \"$1\""

mkdir "instance/data/$run"
mkdir "instance/data/$run/models"
mkdir "instance/data/$run/sgf"
mkdir "instance/data/$run/eval"
mkdir -p "instance/eval/$run"
mkdir -p "instance/policy/$run"
mkdir -p "instance/pv/$run"

