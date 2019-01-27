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
import subprocess
from subprocess import PIPE

import sqlite3
from tqdm import tqdm

from web.cloudygo import CloudyGo

ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
INSTANCE_PATH = os.path.join(ROOT_DIR, 'instance')
DATABASE_PATH = os.path.join(INSTANCE_PATH, 'clouds.db')

BUCKET = CloudyGo.DEFAULT_BUCKET
MODEL_PATH = os.path.join(ROOT_DIR, 'instance', 'data', BUCKET, 'models')

SUFFIX = "_converted.txt.gz"

if not os.path.isdir(MODEL_PATH):
    print ("models dir doesn't exist for", BUCKET)
    sys.exit(1)

#### DB STUFF ####

db = sqlite3.connect(DATABASE_PATH)
db.row_factory = sqlite3.Row

model_range = CloudyGo.bucket_model_range(BUCKET)
cur = db.execute('SELECT model_id_1 % 1000000 '
                 'FROM eval_models '
                 'WHERE model_id_2 == 0 AND model_id_1 BETWEEN ? and ? '
                 'ORDER BY rankings desc '
                 'LIMIT 2',
                 model_range)

best_models = [r[0] for r in cur.fetchall()]
print("Best Models:", best_models)


files = os.listdir(MODEL_PATH)
models = [f.replace('.meta', '') for f in files if f.endswith('.meta')]
converted = [f.replace(SUFFIX, '') for f in files if f.endswith(SUFFIX)]
print("Converted:", converted)

for model in tqdm(sorted(models)):
    num = int(model.split('-')[0])
    if num == 0 or model in converted:
        continue
    if num % 100 == 0 or num in best_models:
        model_path = os.path.join(MODEL_PATH, model)

        lz_path = os.path.join(ROOT_DIR, '..', 'leela-zero', 'training',
                               'minigo', 'convert_minigo.py')
        print (num, model)
        output = subprocess.run([lz_path, model_path],
            stdout=PIPE, stderr=PIPE)
        stdout = output.stdout.decode('utf-8')
        print (output)
        print (stdout)
        print ()
