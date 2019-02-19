#!/usr/bin/python3
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
import os
import re
import urllib.request

BUCKET = 'leela-zero-v4'
BUCKET_NUM = 24

URL = 'http://zero.sjeng.org'
MODEL_DIR = os.path.join('instance', 'data', BUCKET, 'models')
FILE = os.path.join('instance', 'data', BUCKET, 'zero-sjeng-org.html')
INSERTS = os.path.join('instance', 'data', BUCKET, 'inserts.csv')
DOWNLOADER = os.path.join('instance', 'data', BUCKET, 'download.sh')

INSERT = ','.join(['{}'] * 11)

#urllib.request.urlretrieve(URL, FILE)
with open(FILE) as f:
    data = f.read()

with open(INSERTS, 'w') as inserts, open(DOWNLOADER, 'w') as downloader:
    for num in range(1000):
        # check if we can find num in the model table
        match = re.search('<tr>(<td>{}</td>.*)</tr>'.format(num), data)
        if not match:
            print('Did not find {} stopping'.format(num))
            break

        parts = match.group(1).replace('</td>', '').split('<td>')
        assert parts[1] == str(num), parts

        full_name = re.search(r'([0-9a-f]{64})', parts[3]).group()
        name = parts[3][-12:-4]
        games = int(parts[6])
        date = parts[2]
        date = datetime.datetime.strptime(date, '%Y-%m-%d %H:%M')
        epoch = int(date.strftime('%s'))
        network_size = parts[4]
        network_blocks = int(network_size.split('x')[0])

        display_name = 'LZ{}_{}'.format(num, name)

        fname = os.path.join(MODEL_DIR, full_name)
        if not os.path.exists(fname):
            open(fname, 'a').close()
            os.utime(fname, (epoch, epoch))

        display_name = 'LZ{}_{}'.format(num, name)
        # a models row
        row = INSERT.format(
            BUCKET_NUM * 10 ** 6 + num,
            display_name,   # display_name
            display_name,   # name in models dir
            full_name,      # name in sgf files
            BUCKET, num,
            epoch, epoch,
            network_blocks, # training_time_m (being abused)
            games,
            0, # num_stats_games
            0) # num_eval_games
        inserts.write(row + '\n')

        model_path = 'models/' + display_name
        download_command = (
            '[ -f {} ] || (wget {}/networks/{}.gz -O {} && sleep 10)'.format(
                model_path, URL, full_name, model_path))
        downloader.write(download_command + '\n')
        print(row)


commands = [
    ".mode csv",
    ".import instance/data/" + BUCKET + "/inserts.csv models",
]
print()
print('sqlite3 instance/clouds.db', ' '.join(map('"{}"'.format, commands)))
