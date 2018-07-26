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

import bisect
import os
import sqlite3
import time

from tqdm import tqdm

from web.cloudygo import CloudyGo

ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
INSTANCE_PATH = os.path.join(ROOT_DIR, 'instance')
DATABASE_PATH = os.path.join(INSTANCE_PATH, 'clouds.db')

BOARD_SIZE = 19
BUCKET = 'v9-19x19'

#### DB STUFF ####

db = sqlite3.connect(DATABASE_PATH)
db.row_factory = sqlite3.Row

def query_db(query, args=()):
    cur = db.execute(query, args)
    data = list(map(tuple, cur.fetchall()))
    cur.close()
    return data

model_range = CloudyGo.bucket_model_range(BUCKET)
model_creation = query_db('SELECT model_id, creation '
                          'FROM models WHERE bucket = ?', (BUCKET,))
models, creation_times = zip(*model_creation)

T0 = time.time()

game_data = query_db('SELECT game_num, filename, model_id '
                     'FROM games2 WHERE model_id BETWEEN ? AND ?', model_range)
equal = 0
results = []
for game_num, filename, model_id in tqdm(game_data, unit="game"):
    # Assume the game took ~40 minutes
    game_time = int(filename.split('-', 1)[0]) - 40 * 60
    assert 1520000000 < game_time < 1540000000, game_time
    new_i = bisect.bisect(creation_times, game_time, 1)
    assert new_i >= 0
    new_m = models[new_i - 1] # played by an existing model

    if new_m != model_id:
        results.append((new_m, game_num))
    else:
        equal += 1

T1 = time.time()
print('Model Id Guess took: {:.1f} seconds'.format(T1 - T0))

print("\t", results[:3], results[-3:])

cur = db.executemany(
    'UPDATE games2 SET model_id = ? WHERE game_num = ?',
    results)
db.commit()

T2 = time.time()

print('{} == {} rows updated, {} unchanged'.format(
    len(results), cur.rowcount, equal))
print('Update took: {:.1f} seconds'.format(T2 - T1))
