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

import functools
import os
import sqlite3
import time

from tqdm import tqdm

from web.cloudygo import CloudyGo

ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
DATABASE_PATH = os.path.join(ROOT_DIR, 'instance', 'clouds.db')

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
model_ids, creation_times = zip(*model_creation)
model_guesser = functools.partial(
    CloudyGo._model_guesser,
    model_mtimes=creation_times,
    model_ids=model_ids)

T0 = time.time()

game_data = query_db('SELECT timestamp, game_num, filename, model_id '
                     'FROM games WHERE model_id BETWEEN ? AND ?', model_range)
equal = 0
results = []
for ts, game_num, filename, model_id in tqdm(game_data, unit="game"):
    guess_id = model_guesser(filename)

    if guess_id == model_id:
        equal += 1
    else:
        results.append((guess_id, ts, game_num))

T1 = time.time()
print('Model Id Guess took: {:.1f} seconds'.format(T1 - T0))

print("\t", results[:3], results[-3:])

cur = db.executemany(
    'UPDATE games SET model_id = ? WHERE timestamp = ? AND game_num = ?',
    results)
db.commit()

T2 = time.time()

print('{} == {} rows updated, {} unchanged'.format(
    len(results), cur.rowcount, equal))
print('Update took: {:.1f} seconds'.format(T2 - T1))
