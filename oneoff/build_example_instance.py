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
import re
import shutil

from tqdm import tqdm

GAMES = 25

#RUN = 'v3-9x9'
#MODELS = range(290, 310+1)

RUN = 'v5-19x19'
MODELS = range(100, 110+1)

#RUN = 'v7-19x19'
#MODELS = range(180, 190+1)

ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
INSTANCE_DIR = os.path.join(ROOT_DIR, 'instance')
DEVEL_DIR = os.path.join(ROOT_DIR, 'devel_instance')


if not os.path.exists(DEVEL_DIR):
    print(DEVEL_DIR, "does not exist")
    sys.exit(2)

# Source, Dest
RUN_DIR = (os.path.join(INSTANCE_DIR, 'data', RUN),
           os.path.join(DEVEL_DIR, 'data', RUN))
if not os.path.exists(os.path.join(RUN_DIR[0])):
    print("Run dir ({}) does not exists".format(RUN))
    sys.exit(2)

if os.path.exists(os.path.join(RUN_DIR[1])):
    print("Devel run dir ({}) already exists".format(RUN))
    sys.exit(2)

os.mkdir(RUN_DIR[1])

def copy_if_match(A, B, files=None, match_model=False):
    # Copy files (default: all) from dir A to dir B
    if files is None and match_model is None:
        assert False, "Use shutil.copytree"

    if not os.path.exists(A):
        print ("{} Does not exist, skipping".format(A))
        return

    common = os.path.commonprefix([A, B])
    start = len(common)
    print("Moving some of {} | {} to {}".format(
        common, A[start:], B[start:]))

    if not os.path.exists(B):
        os.mkdir(B)

    if not files:
        files = os.listdir(A)

    for f in tqdm(files):
        path = os.path.join(A, f)

        if match_model:
            raw = re.split(r'[._-]+', os.path.basename(path))
            nums = [int(part) for part in raw if part.isnumeric()]
            for num in nums:
                if num in MODELS:
                    break
            else:
                continue

        shutil.copy2(path, os.path.join(B, f))


# Copy all eval games (which match a model num)
def eval_match_model_both(e):
    parts = re.split(r'[._-]+', os.path.basename(e))
    return all(not p.isnumeric() or len(p) != 6 or int(p) in MODELS
        for p in parts)

EVAL_DIR = tuple(os.path.join(run, "eval") for run in RUN_DIR)
eval_files = filter(eval_match_model_both, os.listdir(EVAL_DIR[0]))
copy_if_match(EVAL_DIR[0], EVAL_DIR[1], files=eval_files)
print("eval copied")


# Model dir
MODEL_DIR = tuple(os.path.join(run, "models") for run in RUN_DIR)
os.mkdir(MODEL_DIR[1])
for f in os.listdir(MODEL_DIR[0]):
    try:
        m = int(f.split('-')[0])
        if m in MODELS:
            new_f = os.path.join(MODEL_DIR[1], f)
            with open(new_f, 'w'):
                pass
            # maybe should set utime
    except:
        pass
print("models copied")


# SGF dir
SGF_DIR = tuple(os.path.join(run, "sgf") for run in RUN_DIR)
os.mkdir(SGF_DIR[1])
for m in sorted(os.listdir(SGF_DIR[0])):
    num = int(m.split('-')[0])
    if num not in MODELS:
        continue

    model_dir = tuple(os.path.join(sgf, m) for sgf in SGF_DIR)
    os.mkdir(model_dir[1])

    full_dir = [os.path.join(model, 'full') for model in model_dir]
    clean_dir = [os.path.join(model, 'clean') for model in model_dir]
    if not os.path.exists(full_dir[0]):
        continue

    games = os.listdir(full_dir[0])
    games = random.sample(games, min(len(games), GAMES))
    copy_if_match(full_dir[0], full_dir[1], files=games)
    copy_if_match(clean_dir[0], clean_dir[1], files=games)
print("sgfs copied")

# MiniGo/oneoff dirs
for oneoff in ["eval", "policy", "positions", "pv", "eval"]:
    match_model = oneoff in ("policy", "pv")

    src = os.path.join(INSTANCE_DIR, oneoff, RUN)
    dst = os.path.join(DEVEL_DIR, oneoff, RUN)
    copy_if_match(src, dst, match_model=match_model)
