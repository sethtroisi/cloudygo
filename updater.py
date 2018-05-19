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

import datetime
import glob
import os
import sqlite3
import sys
import time

from multiprocessing import Pool

from web import sgf_utils
from web.cloudygo import CloudyGo


# get this script location to help with running in cron
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
INSTANCE_PATH = os.path.join(ROOT_DIR, 'instance')
LOCAL_DATA_DIR = os.path.join(INSTANCE_PATH, 'data')
DATABASE_PATH = os.path.join(INSTANCE_PATH, 'clouds.db')

CURRENT_BUCKET = 'v7-19x19'
#CURRENT_BUCKET = 'v3-9x9'
BUCKETS = ["v3-9x9", "v5-19x19", "v7-19x19"]


def setup():
    #### DB STUFF ####

    db = sqlite3.connect(DATABASE_PATH)
    db.row_factory = sqlite3.Row


    #### CloudyGo ####

    print("Running updater:", datetime.datetime.now())
    print("Setting up Update Cloudy")
    cloudy = CloudyGo(
        INSTANCE_PATH,
        LOCAL_DATA_DIR,
        lambda: db,
        None, # cache
        Pool(3)
    )
    return cloudy


def update_position_eval(cloudy, bucket, group):
    eval_paths = glob.glob(os.path.join(INSTANCE_PATH, group, bucket, '*'))

    models = cloudy.get_models(bucket)
    model_ids = { model[4] : model[0] for model in models }

    print ("{}: Updating {} {} position evals ({:.2f}/model)".format(
        bucket, group, len(eval_paths), len(eval_paths) / len(models)))

    updates = 0
    for eval_path in eval_paths:
        assert eval_path.endswith('.csv'), eval_path
        raw_name, model_num = os.path.basename(eval_path[:-4]).rsplit('-', 1)
        assert raw_name.startswith("heatmap-") or raw_name.startswith("pv-")
        name = raw_name.split('-', 1)[1]

        model_id = model_ids.get(int(model_num), None)
        if model_id:
            # TODO(sethtroisi): Don't update every time?
            cloudy.update_position_eval(
                eval_path, bucket, model_id, group, name)
            updates += 1

    return updates

def update_position_setups(cloudy, bucket):
    # Refresh the policy/pv setups in db
    position_paths = glob.glob(os.path.join(
        INSTANCE_PATH, "positions", bucket, "*.sgf"))

    position_setups = []
    for position_path in position_paths:
        position_file = os.path.basename(position_path)
        position_name = position_file.replace(".sgf", "")
        with open(position_path) as data:
            sgf_data = data.read().replace("\n", "")
            position_setups.append((bucket, position_name, sgf_data))

    db = cloudy.db()
    db.execute("DELETE from position_setups where bucket = ?", (bucket,))
    cloudy.insert_rows_db("position_setups", position_setups)
    db.commit()

    return len(position_setups)


def update_models_games(cloudy, bucket):
    updates = 0;
    print ("{}: Updating Models and Games".format(bucket))

    # Setup models if they don't exist
    cloudy.update_models(bucket, partial=True)

    models = cloudy.get_models(bucket)
    if len(models) == 0:
        return 0

    # Note when importing a new DB consider lowing and doing multiple updates.
    inserts = 5000
    for stats in [False, True]:
        print ("\tupdating_games({}, {})".format(stats, inserts))
        count, status = cloudy.update_games(bucket, stats, inserts, 0)
        updates += count

    if updates > 0:
        # Sync models with new data
        cloudy.update_models(bucket, partial=False)
    return updates


if __name__ == "__main__":
    T0 = time.time()

    cloudy = setup()

    updates = 0

    if len(sys.argv) == 3 and sys.argv[1] == "models":
        for bucket in BUCKETS:
            updates += cloudy.update_models(
                bucket,
                partial=(sys.argv[2] != "False"))

    if len(sys.argv) == 1 or "games" in sys.argv:
        updates += update_models_games(cloudy, CURRENT_BUCKET)

    if len(sys.argv) == 1 or "eval_games" == sys.argv[1]:
        buckets = [CURRENT_BUCKET] + sys.argv[2:]
        for bucket in buckets:
            updates += cloudy.update_eval_games(bucket)
            updates += cloudy.update_eval_models(bucket)

    if len(sys.argv) == 1 or "position_evals" in sys.argv:
        buckets = [CURRENT_BUCKET]
        for bucket in buckets:
           updates += update_position_setups(cloudy, bucket)
           for group in ["policy", "pv"]:
                updates += update_position_eval(cloudy, bucket, group)

    T1 = time.time()
    delta = T1 - T0
    print ("Updater took: {:.1f} seconds for {} updates = {:.1f}/second".format(
        delta, updates, updates / delta))
