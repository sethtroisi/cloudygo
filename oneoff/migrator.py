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

T0 = time.time()

fields = [
    'game_num',
    'model_id', 'filename',
    'black_won', 'result', 'result_margin',
    'num_moves',
    'early_moves', 'early_moves_canonical',
    'has_stats',
    'number_of_visits_black', 'number_of_visits_white',
    'number_of_visits_early_black', 'number_of_visits_early_white',
    'unluckiness_black', 'unluckiness_white',
    'resign_threshold', 'bleakest_eval_black', 'bleakest_eval_white',
]

#### DB STUFF ####

db = sqlite3.connect(DATABASE_PATH)
db.row_factory = sqlite3.Row

cur = db.execute('SELECT model_id, bucket FROM models')
model_buckets = dict(map(tuple, cur.fetchall()))

cur = db.execute('SELECT model_id, COUNT(*) FROM games2 group by 1')
#                 ' WHERE model_id = 71000363 group by 1')
original_counts = dict(map(tuple, cur.fetchall()))

cur = db.execute('SELECT model_id, COUNT(*) FROM games group by 1 ')
#                 ' WHERE model_id = 71000363 group by 1')
converted_counts = dict(map(tuple, cur.fetchall()))

equal = 0
for k in list(model_buckets.keys()):
    if converted_counts.get(k, 0) == original_counts.get(k, 0):
        equal += 1
        model_buckets.pop(k)
print ("{} already converted".format(equal))


updated = 0
for model_id in tqdm(model_buckets):
    original  = original_counts.get(model_id, 0)
    converted = converted_counts.get(model_id, 0)

    print("model_id: {:<10d}  {} games2, {} games".format(
        model_id, original, converted))

    assert original > converted, (model_id, original, converted)

    cur = db.execute(
        'SELECT ' + ', '.join(fields) + ' FROM games2 WHERE model_id = ?',
        (model_id,))
    rows = list(map(list, cur.fetchall()))
    assert len(rows) == original, (model_id, len(rows), original)

    '''
    cur = db.execute(
        'SELECT timestamp, game_num FROM games WHERE model_id = ?',
        (model_id,))
    existing = set(map(tuple, cur.fetchall()))
    print ("\texisting: ", len(existing))
    '''

    new_games = []
    for game in rows:
        if 13000000 <= game[1] <= 13999999: # LZ
            timestamp = 1
        else:
            filename = game[2]
            assert filename.startswith('15')

            if 'tpu-player' in filename:
                assert filename.endswith('.sgf')
                parts = filename[:-4].split('-')
                timestamp = parts[0]
                pod = parts[-2]
                pod_num = int(parts[-1])
                assert 0 <= pod_num <= 99
            else:
                timestamp, name = filename.split('-', 1)
                pod = name.split('-')[-1][:-4]
                pod_num = 0

            timestamp = int(timestamp)
            assert 1510000000 <= timestamp <= 1540000000
            bucket_num = model_id // 10 ** 6
            game_num = 10000 * int(pod, 36) + 100 * pod_num + bucket_num

            game[0] = game_num

        game[7] = ';'.join(game[7].split(';')[:2])
        game.insert(0, timestamp)
        #if tuple(game[:2]) in existing:
        #    continue

        new_games.append(game)

    print ("\tnew games:", len(new_games))

    '''
    if model_id == 71000363:
        from collections import Counter
        c = Counter(tuple(n[:2]) for n in new_games)
        for k, v in c.items():
            if v > 1:
                print (k)
                for n in new_games:
                    if tuple(n[:2]) == k:
                        print("\t", n)
    '''

    updated += len(new_games)
    values = '({})'.format(','.join(['?'] * (len(fields) + 1)))
    query = 'INSERT INTO games VALUES ' + values
    cur = db.executemany(query, new_games)
    db.commit()

T1 = time.time()
print('{} rows updated'.format(updated))
print('Update took: {:.1f} seconds'.format(T1 - T0))
