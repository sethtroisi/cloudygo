#!/usr/bin/env python3
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

import sys
sys.path.insert(0, '.')

import os
import random
import shutil

from tqdm import tqdm

CLEAN_GAMES = 500
DEBUG_GAMES = 10
MODELS = 150

RUN = 'v7-19x19'


ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
INSTANCE_DIR = os.path.join(ROOT_DIR, 'instance')
DEVEL_DIR = os.path.join(ROOT_DIR, 'devel_instance')

if not os.path.exists(DEVEL_DIR):
    print(DEVEL_DIR, "does not exist")
    sys.exit(2)

RUN_DIR = (os.path.join(INSTANCE_DIR, 'data', RUN),
           os.path.join(DEVEL_DIR, 'data', RUN))
if not os.path.exists(os.path.join(RUN_DIR[0])):
    print("Run dir ({}) does not exists".format(RUN))
    sys.exit(2)

if os.path.exists(os.path.join(RUN_DIR[1])):
    print("Devel run dir ({}) already exists".format(RUN))
    sys.exit(2)

os.mkdir(RUN_DIR[1])

# Copy all eval games
EVAL_DIR = tuple(os.path.join(run, "eval") for run in RUN_DIR)
shutil.copytree(EVAL_DIR[0], EVAL_DIR[1])
print("eval copied")

# Model dir
MODEL_DIR = tuple(os.path.join(run, "models") for run in RUN_DIR)
os.mkdir(MODEL_DIR[1])
for f in os.listdir(MODEL_DIR[0]):
    try:
        m = int(f.split('-')[0])
        if m <= MODELS:
            new_f = os.path.join(MODEL_DIR[1], f)
            with open(new_f, 'w'):
                pass
    except:
        pass
print("models copied")

SGF_DIR = tuple(os.path.join(run, "sgf") for run in RUN_DIR)
os.mkdir(SGF_DIR[1])
print(SGF_DIR[0])
for m in os.listdir(SGF_DIR[0]):
    num = int(m.split('-')[0])
    if num > MODELS:
        continue
    print("model", m)

    model_dir = tuple(os.path.join(sgf, m) for sgf in SGF_DIR)
    os.mkdir(model_dir[1])
    for game_type, count in [('clean', CLEAN_GAMES), ('full', DEBUG_GAMES)]:
        game_folder = [os.path.join(model, game_type) for model in model_dir]
        if not os.path.exists(game_folder[0]):
            continue
        os.mkdir(game_folder[1])
        games = os.listdir(game_folder[0])
        games = random.sample(games, min(len(games), count))
        for game in tqdm(games):
            shutil.copy2(os.path.join(game_folder[0], game), game_folder[1])
print("sgfs copied")
