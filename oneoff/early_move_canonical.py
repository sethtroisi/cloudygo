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

import os
import sqlite3
import time

from tqdm import tqdm

from web import sgf_utils

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
INSTANCE_PATH = os.path.join(ROOT_DIR, 'instance')
DATABASE_PATH = os.path.join(INSTANCE_PATH, 'clouds.db')

BOARD_SIZE = 19

T0 = time.time()

#### DB STUFF ####

db = sqlite3.connect(DATABASE_PATH)
db.row_factory = sqlite3.Row

cur = db.execute('SELECT model_id, bucket FROM models')
model_buckets = dict((map(tuple, cur.fetchall())))

cur = db.execute(
    'SELECT game_num, model_id, early_moves, early_moves_canonical FROM games2')
rows = list(map(tuple, cur.fetchall()))
cur.close()

####

print('Got {} games'.format(len(rows)))
equal = 0
mismatch = 0
results = []
for game_num, model_id, raw_moves, saved_canonical in tqdm(rows, unit="game"):
    bucket = model_buckets[model_id]
    board_size = 9 if '9x9' in bucket else 19

    canonical = sgf_utils.canonical_moves(board_size, raw_moves)
    if canonical == raw_moves:
        equal += 1
    if canonical != saved_canonical:
        if saved_canonical not in (None, ""):
            mismatch += 1
#            print ("{} => {} did not match saved {}".format(
#                raw_moves, canonical, saved_canonical))

        results.append((canonical, game_num))

T1 = time.time()
print('Move canonical(ization) took: {:.1f} seconds'.format(T1 - T0))

cur = db.executemany(
    'UPDATE games SET early_moves_canonical = ? WHERE game_num = ?',
    (results))
db.commit()

T2 = time.time()

print('{} rows updated, {} canonical is same as raw, {} mismatch'.format(
    cur.rowcount, equal, mismatch))
print('Update took: {:.1f} seconds'.format(T2 - T1))
