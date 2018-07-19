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

CURRENT_BUCKET = 'v9-19x19'
#CURRENT_BUCKET = 'v7-19x19'
#CURRENT_BUCKET = 'v5-19x19'
#CURRENT_BUCKET = 'v3-9x9'
#CURRENT_BUCKET = 'leela-zero-v1'

# Note when importing a new DB consider lowing
# for an initial pass to make sure everything is okayw
MAX_INSERTS = 50000


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
        None,  # cache
        Pool(3)
    )
    return cloudy


def update_position_eval(cloudy, bucket, group):
    position_paths = glob.glob(os.path.join(INSTANCE_PATH, group, bucket, '*'))

    models = cloudy.get_models(bucket)
    model_ids = {model[4]: model[0] for model in models}

    print("{}: Updating {} {} position evals ({:.2f}/model)".format(
        bucket, group, len(position_paths), len(position_paths) / len(models)))

    updates = 0
    for position_path in position_paths:
        assert position_path.endswith('.csv'), position_path
        position_name = os.path.basename(position_path[:-4])
        raw_name, model_num = position_name.rsplit('-', 1)
        assert raw_name.startswith("heatmap-") or raw_name.startswith("pv-")
        name = raw_name.split('-', 1)[1]

        model_id = model_ids.get(int(model_num), None)
        if model_id:
            # TODO(sethtroisi): Don't update every time?
            cloudy.update_position_eval(
                position_path, bucket, model_id, group, name)
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


def update_games(cloudy, bucket):
    updates = 0
    print("{}: Updating Models and Games".format(bucket))

    # Setup models if they don't exist
    cloudy.update_models(bucket, partial=True)

    models = cloudy.get_models(bucket)
    if len(models) == 0:
        return 0

    print("\tupdating_games({})".format(MAX_INSERTS))
    count = cloudy.update_games(bucket, MAX_INSERTS)
    updates += count

    if updates > 0:
        # Sync models with new data
        count += cloudy.update_models(bucket, partial=False)
    return updates


if __name__ == "__main__":
    T0 = time.time()

    cloudy = setup()

    updates = 0

    arg1 = sys.argv[1] if len(sys.argv) > 1 else ""
    buckets = sys.argv[2:] if len(sys.argv) > 2 else [CURRENT_BUCKET]

    # Note: Models are also updated in update_games.
    if len(sys.argv) > 3 and arg1 == "models":
        for bucket in sys.argv[3:]:
            updates += cloudy.update_models(
                bucket,
                partial=(sys.argv[2] != "False"))

    if len(sys.argv) == 1 or arg1 == "games":
        for bucket in buckets:
            updates += update_games(cloudy, bucket)

    if len(sys.argv) == 1 or arg1 == "eval_games":
        for bucket in buckets:
            updates += cloudy.update_eval_games(bucket)
            updates += cloudy.update_eval_models(bucket)

    if len(sys.argv) == 1 or arg1 == "position_evals":
        for bucket in buckets:
            updates += update_position_setups(cloudy, bucket)
            for group in ["policy", "pv"]:
                updates += update_position_eval(cloudy, bucket, group)

    # Always update names
    cloudy.update_model_names()
    cloudy.update_bucket_ranges(buckets)

    T1 = time.time()
    delta = T1 - T0
    print("Updater took: {:.1f}s for {} updates = {:.1f}/s".format(
        delta, updates, updates / delta))
