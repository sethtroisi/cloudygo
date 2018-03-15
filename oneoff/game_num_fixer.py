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

import sys; sys.path.insert(0, '.')

import os
import sqlite3
import time

from tqdm import tqdm

from web.cloudygo import CloudyGo

ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
INSTANCE_PATH = os.path.join(ROOT_DIR, 'instance')
DATABASE_PATH = os.path.join(INSTANCE_PATH, 'clouds.db')

BOARD_SIZE = 19

#### DB STUFF ####

db = sqlite3.connect(DATABASE_PATH)
db.row_factory = sqlite3.Row

cur = db.execute('SELECT model_id, bucket FROM models')
model_buckets = dict((map(tuple, cur.fetchall())))
bucket_salts = {k:CloudyGo.bucket_salt(v) for k, v in model_buckets.items()}

T0 = time.time()

cur = db.execute('SELECT game_num, filename, model_id FROM games')
rows = list(map(tuple, cur.fetchall()))
cur.close()

####

print ('Got {} games'.format(len(rows)))
equal = 0
mismatch = 0
results = []
for game_num, filename, model_id in tqdm(rows, unit="game"):
    bucket_salt = bucket_salts[model_id]

    canonical = CloudyGo.get_game_num(bucket_salt, filename)
    if canonical != game_num:
        mismatch += 1
        results.append((canonical, game_num))
    else:
        equal += 1

T1 = time.time()
print ('Game Num Fixer took: {:.1f} seconds'.format(T1 - T0))

print ("\t", results[:5])

for table in ('games', 'game_stats'):
    cur = db.executemany(
        'UPDATE ' + table + ' '
        'SET game_num = ? WHERE game_num = ?', (results))
db.commit()

T2 = time.time()

print ('{} rows updated, {} canonical is same as raw, {} mismatch'.format(
    cur.rowcount, equal, mismatch))
print ('Update took: {:.1f} seconds'.format(T2 - T1))
