#!/usr/bin/env bash
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

echo "Buckets:"
sqlite3 instance/clouds.db "select bucket from bucket_model_range order by 1"

echo "Found:"
sqlite3 instance/clouds.db "select * from runs"

echo "Inserting:"
sqlite3 instance/clouds.db """
    INSERT INTO RUNS VALUES
        ('cross-eval', 'cross-eval', 'Unsorted eval game',
            0, 0),
        ('cross-run-eval', 'cross-run-eval', 'Eval games between minigo runs',
            0, 0),
        ('leela-zero', 'leela-zero', 'Leela-Zero (games and eval)',
            0, 0),
        ('leela-zero-eval', 'leela-zero eval', 'Leela-Zero (eval with more cross play)',
            0, 0),
        ('leela-zero-eval-time', 'leela-zero eval with even time', 'Leela-Zero (eval on equal time)',
            0, 0),
        ('KataGo', 'KataGo', 'KataGo g104',
            20, 256);
        ('synced-eval', 'synced-eval', 'Eval games between minigo runs',
            0, 0),
        ('v3-9x9', 'v3', 'Old 9x9 Run',
            10, 32),
        ('v5-19x19', 'v5', 'First 19x19 Run',
            20, 128),
        ('v7-19x19', 'v7', '',
            20, 128),
        ('v9-19x19', 'v9', '',
            20, 256),
        ('v10-19x19', 'v10', '',
            20, 256),
        ('v11-19x19', 'v11', 'Experiment: Q=draw',
            20, 256),
        ('v12-19x19', 'v12', 'Experiment: BatchSize=2',
            20, 256),
        ('v13-19x19', 'v13', 'Human Bootstrap',
            21, 256),
        ('v14-19x19', 'v14', 'Q=loss + Bigtable',
            20, 256),
        ('v15-19x19', 'v15', 'Q=loss',
            20, 256),
        ('v16-19x19', 'v16', 'First 40 block run',
            40, 256),
        ('v17-19x19', 'v17', 'Squeeze and Exicitation',
            20, 256);
"""
