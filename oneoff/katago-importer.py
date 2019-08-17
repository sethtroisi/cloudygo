#!/usr/bin/env bash
#
# Copyright 2019 Google LLC
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

#   create get download.sh
# $ cat zips.txt  | grep -o 'b.*' | xargs -I {} echo '[ -f "{}" ] || (wget https://d3dndmfyhecmj0.cloudfront.net/g104/selfplay/{} && sleep 3)' > download.sh
# $ chmod a+x download.sh
# $ cd zips
# $ ../download.sh
#   unextract zips
# $ ls *.zip | xargs -I {} unzip {} -x "*/tdata/*" "*/vdata/*"


# TODO KataGo-104? KataGo-g65? KataGo-V1? KataGo-2019_1?

import os
import re
import zlib
from collections import defaultdict, Counter

from tqdm import tqdm


PB_RE = re.compile('PB\[[^]]*]')


def consistent_hash(string):
    return zlib.adler32(string.encode('utf-8'))


def touch_utime(path, epoch):
    if not os.path.exists(path):
        open(path, 'a').close()
        os.utime(path, (epoch, epoch))


def extract_model_sgfs_to_folders(bucket, bucket_num, dest):
    bucket_dir    = os.path.join('instance', 'data', bucket)
    models_dir    = os.path.join(bucket_dir, 'models')
    inserts_path  = os.path.join(bucket_dir, 'inserts.csv')

    models = []
    game_num = 1
    player_names = defaultdict(Counter)

    for model_name in sorted(os.listdir(os.path.join(bucket_dir, 'zips'))):
        model_name_short = model_name.rsplit('-', 1)[0]
        print(f'model_name: {model_name_short}')

        sgf_dir = os.path.join(bucket_dir, 'zips', model_name, 'sgfs')
        if not os.path.isdir(sgf_dir): continue

        dest_dir = os.path.join(bucket_dir, dest, model_name_short)
        existed = os.path.isdir(dest_dir)
        if not existed: os.mkdir(dest_dir)

        model_games = 0
        for combined_f in os.listdir(sgf_dir):
            assert combined_f.endswith('.sgfs'), combined_f
            combined_f = os.path.join(sgf_dir, combined_f)

            # We set the model time to the mtime of a random .sgfs file.
            epoch = int(os.path.getmtime(combined_f))

            with open(combined_f) as sgf_file:
                for line in tqdm(sgf_file.readlines()):
                    player = PB_RE.search(line)
                    assert player, line
                    player = player.group(0)
                    player_names[model_name_short][player] += 1

                    if not existed:
                        game_path = os.path.join(dest_dir, 'KataGo-{:08d}.sgf'.format(game_num))
                        with open(game_path, 'w') as sgf_f:
                            sgf_f.write(line)
                            game_num += 1
                            model_games += 1

        for player, count in player_names[model_name_short].most_common():
            print(f'\t{player} x{count} sgfs')

        ##### Model Data #####
        full_name = model_name
        name = model_name_short
        print (name)
        network_size = re.search(r'b[0-9]+c', name).group()[1:-1]
        network_blocks = re.search(r'c[0-9]+\-', name).group()[1:-1]
        network_steps = int(re.search(r's[0-9]+$', name).group()[1:-1])

        display_name = 'KataGo-{}'.format(name)

        fpath = os.path.join(models_dir, full_name)
        touch_utime(fpath, epoch)

        dpath = os.path.join(models_dir, display_name)
        touch_utime(dpath, epoch)

        models.append([
            -1,             # model_id
            display_name,   # display_name
            full_name,      # name in models dir
            full_name,      # name in sgf files
            bucket,
            -1,             # num
            epoch, epoch,
            network_blocks, # training_time_m (being abused)
            model_games,
            0, # num_stats_games
            network_steps, # num_eval_games
        ])

    print('\n')
    all_players = sum(player_names.values(), Counter())
    for player, count in all_players.most_common():
        print(f'\t{player} x{count} sgfs')

    print('Sorted')
    for player, count in sorted(all_players.items()):
        print(f'\t{player} x{count} sgfs')

    with open(inserts_path, 'w') as inserts_f:
        models.sort(key=lambda m: m[11])
        for m_i, model in enumerate(models):
            model[0] = bucket_num * 10 ** 6 + m_i
            model[5] = m_i
            model[11] = 0

            row = ','.join(map(str, model))
            inserts_f.write(row + '\n')
            print(row)

    print()
    print('sqlite3 instance/clouds.db',
          '".mode csv"',
          '".import instance/data/' + bucket + '/inserts.csv models"')


if __name__ == '__main__':
    bucket = 'KataGo'
    bucket_num = consistent_hash(bucket) % 100

    print('bucket: {}, num: {}'.format(bucket, bucket_num))
    print()
    print('Self Play')
    extract_model_sgfs_to_folders(bucket, bucket_num, 'sgf')
