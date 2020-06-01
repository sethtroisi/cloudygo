#!/usr/bin/python
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
import re
import time
import zlib
from collections import defaultdict
from datetime import datetime

from . import sgf_utils
from . import utils


class CloudyGo:
    SALT_MULT = 10 ** 6

    CROSS_EVAL_START   = 10000 # offset from SALT_MULT to start
    SPECIAL_EVAL_START = 20000 # offset ...

    SPECIAL_EVAL_NAMES = [
        "GNU Go:3.8",
        "Pachi UCT:12.20",
    ]

    # FAST UPDATE HACK fastness
    FAST_UPDATE_HOURS = 6
    MAX_INSERTS = 200000

    # To avoid storing all full (AKA debug) games, hotlink from the projects
    # Google Cloud Storage bucket. Some runs are in this public bucket, but
    # the current run may not yet be rsynced so we have extra secret path
    # described in the next comment.
    FULL_GAME_CLOUD_BUCKET = 'minigo-pub'

    # AMJ has asked me to keep this a secret.
    SECRET_CLOUD_BUCKET = os.environ.get(
        'SECRET_CLOUD_BUCKET_PREFIX', FULL_GAME_CLOUD_BUCKET)

    DEFAULT_BUCKET = 'v17-19x19'
    LEELA_ID = 'leela-zero'
    KATAGO_ID = 'KataGo'

    # These do special things with PB and PW
    ALL_EVAL_BUCKETS = ['synced-eval', 'cross-run-eval']

    CROSS_EVAL_BUCKET_MODEL = re.compile(
        r'1[0-9]{9}.*(v[0-9]+)-([0-9]*)-(?:vs-)*(v[0-9]*)-([0-9]*)')

    # NOTE: From v9 on sgf folders has timestamp instead of model directories
    # this radically complicates several parts of the update code. Those places
    # should be documented with either MINIGO_TS or MINIGO-HACK.
    MINIGO_TS = ['v9-19x19',  'v10-19x19', 'v11-19x19', 'v12-19x19',
                 'v13-19x19', 'v14-19x19', 'v15-19x19', 'v16-19x19',
                 'v17-19x19', 'v18-19x19', 'v19-19x19', 'v20-19x19']
    # Average length of game in seconds, used to attribute game to previous model.
    MINIGO_GAME_LENGTH = 25 * 60

    MODEL_CKPT = 'model.ckpt-'

    MIN_TS = int(datetime(2017, 6, 1).timestamp())
    MAX_TS = int(datetime.utcnow().timestamp() + 10 * 86400)

    # set by __init__ but treated as constant
    INSTANCE_PATH = None
    DATA_DIR = None


    def __init__(self, instance_path, data_dir, database, cache, pool):
        self.INSTANCE_PATH = instance_path
        self.DATA_DIR = data_dir

        self.db = database
        self.cache = cache
        self.pool = pool

        self.last_cloud_request = 0
        self.storage_clients = {}

    #### PATH UTILS ####

    def data_path(self, bucket):
        return '{}/{}'.format(self.DATA_DIR, bucket)

    def model_path(self, bucket):
        return os.path.join(self.data_path(bucket), 'models')

    def sgf_path(self, bucket):
        return os.path.join(self.data_path(bucket), 'sgf')

    def eval_path(self, bucket):
        return os.path.join(self.data_path(bucket), 'eval')

    def some_model_games(self, bucket, model_id, limit):
        query = 'SELECT filename FROM games WHERE model_id = ? LIMIT ?'
        return [p[0] for p in self.query_db(query, (model_id, limit))]

    #### db STUFF ####

    def query_db(self, query, args=()):
        cur = self.db().execute(query, args)
        rv = cur.fetchall()
        cur.close()
        # TODO sqlite3.Row is amazing, stop being silly here
        return list(map(tuple, rv))

    @staticmethod
    def bucket_condition(bucket):
        min_model = CloudyGo.bucket_salt(bucket)
        assert isinstance(min_model, int), min_model
        return ' (model_id BETWEEN {} AND {}) '.format(
            min_model, min_model + CloudyGo.SALT_MULT)

    def bucket_query_db(
            self, bucket,
            select, table, where,
            group_count, limit=1000, args=()):
        '''Build a sql query

        select is in the form 'SELECT field % something, stat'
        table is just the table name 'game' or 'model_stats'
        where is any conditions on the select
        group_count is a number of terms to GROUP BY AND ORDER BY
        '''
        assert select.startswith('SELECT '), select
        assert where.startswith('WHERE ') or len(where) == 0, where

        if where:
            where = where + ' AND '
        else:
            where = 'WHERE '
        where += CloudyGo.bucket_condition(bucket)

        group_by = ','.join(map(str, range(1, group_count+1)))

        query = '{} FROM {} {} GROUP BY {} ORDER BY {} DESC LIMIT {}'.format(
            select, re.escape(table), where, group_by, group_by, limit)

        return list(reversed(self.query_db(query, args)))

    def insert_rows_db(self, table, rows, allow_existing=False):
        assert re.match('^[a-zA-Z_2]*$', table), table
        if len(rows) > 0:
            values = '({})'.format(','.join(['?'] * len(rows[0])))
            replace_text = 'OR REPLACE' if allow_existing else ''
            query = 'INSERT {} INTO {} VALUES {}'.format(
                replace_text, re.escape(table), values)
            self.db().executemany(query, rows)

    #### MORE UTILS ####

    @staticmethod
    def bucket_to_board_size(bucket):
        return 9 if '9x9' in bucket else 19

    @staticmethod
    def consistent_hash(string):
        return zlib.adler32(string.encode('utf-8'))

    @staticmethod
    def bucket_salt(bucket):
        return CloudyGo.SALT_MULT * (CloudyGo.consistent_hash(bucket) % 100)

    @staticmethod
    def bucket_model_range(bucket):
        bucket_salt = CloudyGo.bucket_salt(bucket)
        return (bucket_salt, bucket_salt + CloudyGo.SALT_MULT - 1)


    @staticmethod
    def get_cloud_bucket(bucket):
        cloud_bucket = CloudyGo.SECRET_CLOUD_BUCKET
        if cloud_bucket != CloudyGo.FULL_GAME_CLOUD_BUCKET:
            # the secret bucket name includes part of the bucket name.
            cloud_bucket += bucket.split('x')[0]
        return cloud_bucket

    @staticmethod
    def get_game_num(bucket_salt, filename):
        # LEELA-HACK, KATAGO-HACK
        if (CloudyGo.LEELA_ID in filename) or (CloudyGo.KATAGO_ID in filename):
            number = filename.rsplit('-', 1)[-1]
            assert number.endswith('.sgf')
            # TODO(sethtroisi): these are generated from sgfsplit
            # come up with a better scheme (hash maybe) for numbering
            return (1, int(number[:-4]))

        # MINIGO-HACK for timestamps
        # TODO: replace this hack with something different
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

        assert bucket_salt % CloudyGo.SALT_MULT == 0, bucket_salt
        bucket_num = bucket_salt // CloudyGo.SALT_MULT

        timestamp = int(timestamp)
        game_num = 10000 * int(pod, 36) + 100 * pod_num + bucket_num
        return (timestamp, game_num)

    @staticmethod
    def time_stamp_age(mtime):
        now = datetime.now()
        was = datetime.fromtimestamp(mtime)
        delta = now - was
        deltaDays = str(delta.days) + ' days '
        deltaHours = str(round(delta.seconds / 3600, 1)) + ' hours ago'

        return [
            was.strftime("%Y-%m-%d %H:%M"),
            (deltaDays if delta.days > 0 else '') + deltaHours
        ]

    def get_run_data(self, bucket):
        runs = self.query_db(
            'SELECT * FROM runs WHERE bucket = ?',
            (bucket,))
        assert len(runs) <= 1, (bucket, runs)
        return runs[0] if runs else []

    def get_models(self, bucket):
        return self.query_db(
            'SELECT * FROM models WHERE bucket = ? ORDER BY model_id',
            (bucket,))

    def get_newest_model_num(self, bucket):
        model_nums = self.query_db(
            'SELECT num FROM models WHERE bucket = ? ORDER BY num DESC LIMIT 1',
            (bucket,))

        # cross-eval does this and other things.
        if len(model_nums) == 0:
            return 1

        assert len(model_nums) > 0, model_nums
        return model_nums[0][0]

    def load_model(self, bucket, model_name):
        if str(model_name).lower() == 'newest':
            model_name = self.get_newest_model_num(bucket)

        model = self.query_db(
            'SELECT * FROM models WHERE '
            '    bucket = ? AND '
            '    (display_name = ? OR raw_name = ? OR num = ?)',
            (bucket, model_name, model_name, model_name))

        stats = tuple()
        if model:
            stats = self.query_db(
                'SELECT * FROM model_stats WHERE model_id = ? '
                'ORDER BY perspective ASC',
                (model[0][0],))

        return (
            model[0] if len(model) > 0 else None,
            stats if len(stats) > 0 else None
        )

    def load_games(self, bucket, filenames):
        bucket_salt = CloudyGo.bucket_salt(bucket)

        games = []
        for filename in filenames:
            game_num = CloudyGo.get_game_num(bucket_salt, filename)
            # NOTE: if table is altered * may return unexpected order

            game = self.query_db(
                'SELECT * FROM games WHERE timestamp = ? AND game_num = ?',
                game_num)
            if len(game) == 1:
                games.append(game[0])
        return games

    def __get_gs_game(self, bucket, model_name, filename, view_type):
        assert 'full' in view_type, view_type


        # Maybe it's worth caching these for, now just globally rate limit
        now = time.time()
        if now - self.last_cloud_request < 1:
            return None
        self.last_cloud_request = now

        # NOTE: needs to be before cloud_bucket clears bucket.
        from google.cloud import storage
        cloud_bucket = CloudyGo.get_cloud_bucket(bucket)
        if bucket not in self.storage_clients:
            client = storage.Client(project="minigo-pub").bucket(cloud_bucket)
            self.storage_clients[bucket] = client

        # MINIGO-HACK
        if bucket in CloudyGo.MINIGO_TS:
            # Take a guess at based on timestamp
            hour_guess = CloudyGo.guess_hour_dir(filename)
            model_name = hour_guess

            path = os.path.join('sgf', 'full', hour_guess, filename)
            if cloud_bucket == CloudyGo.FULL_GAME_CLOUD_BUCKET:
                # MINIGO_PUB has an outer folder of the bucket name
                path = os.path.join(bucket, path)
        else:
            path = os.path.join(bucket, 'sgf', model_name, 'full', filename)

        blob = self.storage_clients[bucket].get_blob(path)
        print("Checking {}: {}".format(filename, blob is not None))
        print(self.storage_clients[bucket], path)
        if not isinstance(blob, storage.Blob):
            return None

        data = blob.download_as_string().decode('utf8')
        return data

    @staticmethod
    def guess_hour_dir(filename):
        file_time = int(filename.split('-', 1)[0])
        assert CloudyGo.MIN_TS < file_time < CloudyGo.MAX_TS, file_time
        dt = datetime.utcfromtimestamp(file_time)
        return dt.strftime("%Y-%m-%d-%H")

    @staticmethod
    def guess_number_dir(filename):
        # leela-zero-vX-00001234.sgf
        end = filename.rsplit('-', 1)[1]
        assert end.endswith('.sgf')
        number = int(end[:-4])
        assert 0 < number < 50000000, number
        # See ../oneoff/leela-all-to-dirs.sh PER_FOLDER
        return str(number - (number % 5000))


    def get_game_data(self, bucket, model_name, filename, view_type):
        # Reconstruct path from filename

        base_path = os.path.join(self.sgf_path(bucket), model_name)
        if view_type == 'eval':
            base_path = os.path.join(self.data_path(bucket))

        file_path = os.path.join(base_path, view_type, filename)

        # NOTE: To avoid making filename longer when it's determinisitic
        # These two directory guesses are needed. If this becomes an issue
        # I may revisit this, store the repeated folder name in the DB
        # and not worry about this.

        if view_type != 'eval' and not os.path.isfile(file_path):
            base_path = os.path.join(self.sgf_path(bucket))

            # MINIGO-HACK
            if bucket in CloudyGo.MINIGO_TS:
                hour_guess = CloudyGo.guess_hour_dir(filename)
                file_path = os.path.join(base_path, view_type,
                                         hour_guess, filename)

            # LEEZA-HACK
            if bucket == CloudyGo.LEELA_ID:
                dir_guess = CloudyGo.guess_number_dir(filename)
                file_path = os.path.join(base_path, dir_guess, filename)

            # KATAGO-HACK
            if CloudyGo.KATAGO_ID in bucket:
                dir_guess = re.sub(r'(KataGo-)?(.*)(-d[0-9]*)?$', r'\2', model_name)
                file_path = os.path.join(base_path, dir_guess, filename)

        base_dir_abs = os.path.abspath(base_path)
        file_path_abs = os.path.abspath(file_path)
        if not file_path_abs.startswith(base_dir_abs) or \
           not file_path_abs.endswith('.sgf'):
            return 'being naughty?', view_type

        data = ''
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                data = f.read()
            return data, view_type

        # NOTE: All clean games are kept but condition could be removed.
        if 'full' in view_type \
                and CloudyGo.LEELA_ID not in bucket \
                and CloudyGo.KATAGO_ID not in bucket \
                and CloudyGo.FULL_GAME_CLOUD_BUCKET:
            data = self.__get_gs_game(bucket, model_name, filename, view_type)
            if data:
                return data, view_type

        # Full games get deleted after X time, fallback to clean
        if 'full' in view_type:
            new_type = view_type.replace('full', 'clean')
            return self.get_game_data(bucket, model_name, filename, new_type)

        return data, view_type


    def _get_existing_games(self, where, args):
        query = ('SELECT timestamp, game_num, has_stats '
                 'FROM games '
                 'WHERE ' + where)
        return {(ts,g_n): h_s for ts, g_n, h_s in self.query_db(query, args)}

    def _get_games_from_model(self, model_id):
        return self._get_existing_games('model_id = ?', model_id)

    def _get_games_from_models(self, model_range):
        return self._get_existing_games(
            'model_id BETWEEN ? and ?',
            model_range)

    def _get_games_from_ts(self, model_range, ts_range):
        return self._get_existing_games(
            'model_id BETWEEN ? and ? AND timestamp BETWEEN ? and ?',
            model_range + ts_range)

    def _get_eval_games(self, bucket):
        model_range = CloudyGo.bucket_model_range(bucket)
        query = ('SELECT eval_num '
                 'FROM eval_games '
                 'WHERE model_id_1 BETWEEN ? AND ?')
        records = self.query_db(query, model_range)
        return set(r[0] for r in records)

    def get_position_sgfs(self, bucket, model_ids=None):
        bucket_salt = CloudyGo.bucket_salt(bucket)

        # Single model_id, two model_ids, all model_ids
        where = 'WHERE cord = -2 AND model_id BETWEEN ? AND ?'
        args = (bucket_salt + 5, bucket_salt + CloudyGo.SALT_MULT)

        if model_ids == None:
            pass
        elif len(model_ids) == 1:
            args = (model_ids[0], model_ids[0] + 1)
        elif len(model_ids) == 2:
            where = 'WHERE cord = -2 AND (model_id == ? or model_id == ?)'
            args = (model_ids[0], model_ids[1])
        else:
            assert False, model_ids

        sgfs = self.query_db(
            'SELECT model_id, name, type, sgf, round(value,3) '
            'FROM position_eval_part ' +
            where,
            args)

        arranged = defaultdict(lambda: defaultdict(
            lambda: defaultdict(lambda: ('', 0))))
        count = 0
        names = set()
        for model_id, name, group, sgf, value in sgfs:
            assert group in ('pv', 'policy'), group
            model_id %= CloudyGo.SALT_MULT

            names.add(name)
            arranged[model_id][name][group] = (sgf, value)
            count += 1

        setups = self.query_db(
            'SELECT name, sgf FROM position_setups WHERE bucket = ?',
            (bucket,))
        setups = {name: sgf_utils.count_moves(sgf) for name, sgf in setups}
        setups["empty"] = 0

        names = sorted(names, key=lambda n: (setups.get(n, 10), name))

        # TODO(sethtroisi): add PV value.

        data = []
        for m in sorted(arranged.keys(), reverse=True):
            rows = []
            for n in names:
                rows.append(
                    (n.replace('_', ''),) +
                    arranged[m][n]['policy'] +
                    arranged[m][n]['pv'] +
                    (setups.get(n, 10),)
                )
            data.append([m] + rows)

        return count, data

    def get_position_eval(self, bucket, model_id, group, name):
        sgf = self.query_db(
            'SELECT sgf '
            'FROM position_eval_part '
            'WHERE model_id = ? AND type = ? AND name = ? ',
            (model_id, group, name))
        assert len(sgf) <= 1, (bucket, model_id, group, name)
        if len(sgf) == 1:
            return sgf[0][0]

    def render_position_eval(
            self, bucket, model_id,
            group, name, data,
            filename=None):
        # Old style
        # data = self.query_db(
        #    'SELECT cord, policy, n '
        #    'FROM position_eval_part '
        #    'WHERE model_id = ? AND type = ? AND name = ? '
        #    'ORDER BY n DESC, policy DESC',
        #    (model_id, group, name))

        board_size = CloudyGo.bucket_to_board_size(bucket)

        # The V stands for 'value' as in 'policy value'
        # I'm very sorry about that future reader.
        high_v = sorted([p for c, p, n in data], reverse=True)
        cutoff = 0.005
        if len(high_v) > 15:
            cutoff = max(cutoff, high_v[15])

        position_nodes = []
        for move_num, (cord, policy, n) in enumerate(data, 1):
            if 0 <= cord < board_size*board_size and (n > 0 or policy > cutoff):
                j, i = divmod(cord, board_size)
                if n > 0:
                    value = move_num
                else:
                    value = round(100 * policy, 1)

                position_nodes.append((
                    sgf_utils.ij_to_cord(board_size, (i, j)),
                    value))

        # Look up the position setup
        position_setup = self.query_db(
            'SELECT sgf FROM position_setups '
            'WHERE bucket = ? AND name = ? '
            'LIMIT 1;',
            (bucket, name))

        position_setup = position_setup[0][0] if len(position_setup) else ''
        # SGF parsing is hard, so this is hardcoded (for now)
        # skip to second ';' and include everything up to last ')'
        if ';' in position_setup:
            position_setup = ';' + position_setup.split(';', 2)[-1][:-1]

        is_pv = group == 'pv'

        return sgf_utils.board_png(
            board_size,
            position_setup,
            position_nodes,
            filename,
            include_move=is_pv or (name == 'empty'),
            is_pv=is_pv,
        )

    def get_favorite_openings(self, model_id, num_games):
        favorite_openings = self.query_db(
            'SELECT SUBSTR(early_moves_canonical,'
            '              0, instr(early_moves_canonical, ";")),'
            '       count(*)'
            'FROM games WHERE model_id = ? '
            'GROUP BY 1 ORDER BY 2 DESC LIMIT 16',
            (model_id,))

        return [(move, round(100 * count / num_games, 1))
                for move, count in favorite_openings
                if move and num_games and 500 * count >= num_games]

    #### PAGES ####

    def get_model_names(self, model_range):
        raw_names = self.query_db(
            'SELECT model_id, name, source FROM name_to_model_id '
            'WHERE model_id BETWEEN ? AND ? '
            'ORDER by source ', # 'model' before 'sgf'
            model_range)

        # TODO: make this generic?
        display_names = self.query_db(
            'SELECT model_id, display_name FROM models '
            'WHERE model_id BETWEEN ? AND ? ',
            model_range)

        num_to_name = {}
        for model_id, display_name in display_names:
            num_to_name[model_id] = display_name

        for model_id, name, source in raw_names:
            if name.startswith(CloudyGo.MODEL_CKPT):
                continue

            if "ELF" in name:
                continue

            existing = num_to_name.get(model_id)
            if existing == name:
                continue

            if existing and source == 'model':
                continue

            if existing:
                # NOTE(sethtroisi): Models having multiple names complicates
                # things, e.g. eval page.
                print("ERROR: {}, multiple names: {}, {}".format(
                    model_id, name, num_to_name[model_id]))

            num_to_name[model_id] = name

        for model_id, name, source in raw_names:
            if model_id in num_to_name:
                continue

            # if no name other than model.ckpt, use that name
            if name.startswith(CloudyGo.MODEL_CKPT):
                if model_id % CloudyGo.SALT_MULT < CloudyGo.CROSS_EVAL_START:
                    print("error: {} only had ckpt name ({})".format(
                        model_id, name))
                num_to_name[model_id] = name

        return num_to_name
