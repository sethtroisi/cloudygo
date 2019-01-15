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

import sqlite3
import sys
import time


def main(argv):
    if len(argv) != 2:
        print("Usage: delete_bucket.py bucket")
        sys.exit(1)

    db = sqlite3.connect("instance/clouds.db")
    db.row_factory = sqlite3.Row

    def query_db(query, args=()):
        cur = db.execute(query, args)
        rv = cur.fetchall()
        cur.close()
        return list(map(tuple, rv))

    buckets = query_db("select * from bucket_model_range")
    names = [b[0] for b in buckets]
    rm_bucket = argv[1]
    print("Buckets:", ", ".join(names))
    print("Removing:", rm_bucket)
    print()
    assert rm_bucket in names
    m_low, m_high = min([b[1:] for b in buckets if b[0] == rm_bucket])

    tables = query_db("select name from sqlite_master where type = 'table'")
    tables = [t[0] for t in tables]
    print("Tables:", ", ".join(tables))

    tables_model = ["model_stats", "models", "position_eval_part", "name_to_model_id", "games"]
    tables_model_1 = ["eval_models", "eval_games", "bucket_model_range"]

    total_count = 0
    for table in tables_model + tables_model_1:
        query = "select count(*) from {} where {} between ? and ?".format(
            table,
            "model_id" if table in tables_model else "model_id_1")
        count = query_db(query, (m_low, m_high))[0][0]
        if count > 0:
            total_count += count
            print(table, count, total_count)
    print()

    if total_count > 0:
        test = input("type: \"delete_{}\" to delete: ".format(rm_bucket))
        if test == "delete_" + rm_bucket:
            print("Deleting in 3s")
            time.sleep(3)
            print()
            for table in tables_model + tables_model_1:
                query = "delete from {} where {} between ? and ?".format(
                    table,
                    "model_id" if table in tables_model else "model_id_1")
                query_db(query, (m_low, m_high))
            print("Deleted")


if __name__ == "__main__":
    main(sys.argv)
