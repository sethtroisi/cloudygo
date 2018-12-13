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

import bisect
import functools
import glob
import hashlib
import itertools
import math
import os
import re
import time
import zlib
from collections import Counter, defaultdict, OrderedDict
from datetime import datetime, timezone

import choix
import numpy as np
from tqdm import tqdm

from . import sgf_utils
from . import utils


class CloudyGo:
    SALT_MULT = 10 ** 6

    CROSS_EVAL_START = 2500  # offset from SALT_MULT to start

    # FAST UPDATE HACK fastness
    FAST_UPDATE_HOURS = 6
    MAX_INSERTS = 50000

    # set by __init__ but treated as constant
    INSTANCE_PATH = None
    DATA_DIR = None

    # To avoid storing all full (AKA debug) games, hotlink from the projects
    # Google Cloud Storage bucket. Some runs are in this public bucket, but
    # the current run may not yet be rsynced so we have extra secret path
    # described in the next comment.
    FULL_GAME_CLOUD_BUCKET = 'minigo-pub'

    # AMJ has asked me to keep this a secret.
    SECRET_CLOUD_BUCKET = os.environ.get(
        'SECRET_CLOUD_BUCKET_PREFIX', FULL_GAME_CLOUD_BUCKET)

    DEFAULT_BUCKET = 'v14-19x19'
    LEELA_ID = 'leela-zero'

    # NOTE: From v9 on sgf folders has timestamp instead of model directories
    # this radically complicates several parts of the update code. Those places
    # should be documented with either MINIGO_TS or MINIGO-HACK.
    MINIGO_TS = ['v9-19x19',  'v10-19x19', 'v11-19x19', 'v12-19x19',
                 'v13-19x19', 'v14-19x19', 'v15-19x19', 'v16-19x19']
    MINIGO_GAME_LENGTH = 25 * 60


    MODEL_CKPT = 'model.ckpt-'

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

    def all_games(self, bucket, model, game_type='full'):
        # LEELA-HACK
        if CloudyGo.LEELA_ID in bucket:
            game_type = 'clean'

        if bucket in CloudyGo.MINIGO_TS:
            assert False, bucket

        # NOTE: An older version of cloudygo would load games in two passes
        # Parsing clean then full games this gave some flexibility but at
        # the cost of overall speed now the code tries to do both parses
        # in one pass falling back to simple_parse when debug information is
        # not present.
        # TODO: Support fallback to clean dir
        path = os.path.join(self.sgf_path(bucket), model, game_type)
        if not os.path.exists(path):
            return []
        return glob.glob(os.path.join(path, '*.sgf'))

    #### db STUFF ####

    def query_db(self, query, args=()):
        cur = self.db().execute(query, args)
        rv = cur.fetchall()
        cur.close()
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
    def get_game_num(bucket_salt, filename):
        # LEELA-HACK
        if CloudyGo.LEELA_ID in filename:
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
    def get_eval_parts(filename):
        assert filename.endswith('.sgf'), filename

        # TODO(sethtroisi): What is a better way to determine if this is part
        # of a run (e.g. LZ, MG) or a test eval dir?

        # MG: 1527290396-000241-archer-vs-000262-ship-long-0.sgf
        # LZ: 000002-88-fast-vs-18-fast-202.sgf
        is_run = re.match(
            r'^[0-9]+-[0-9]+-[a-z-]+-vs-[0-9]+-[a-z-]+-[0-9]+\.sgf$',
            filename)

        if is_run:
            # make sure no dir_eval games end up here
            raw = re.split(r'[._-]+', filename)
            nums = [int(part) for part in raw if part.isnumeric()]
            assert len(nums) == 4, '{} => {}'.format(filename, raw)
            SEP = 1000
            assert max(nums[1:]) < SEP, nums
            multed = sum(num * SEP ** i for i, num in enumerate(nums[::-1]))
            return [multed, nums[1], nums[2]]

        MAX_EVAL_NUM = 2 ** 60
        num = int(hashlib.md5(filename.encode()).hexdigest(), 16)
        return [num % MAX_EVAL_NUM, 0, 0]

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
            'SELECT * FROM models WHERE bucket = ? AND (raw_name = ? or num = ?)',
            (bucket, model_name, model_name))

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

    def __get_gs_game(self, bucket, model, filename, view_type):
        assert 'full' in view_type, view_type


        # Maybe it's worth caching these for, now just globally rate limit
        now = time.time()
        if now - self.last_cloud_request < 1:
            return None
        self.last_cloud_request = now

        # NOTE: needs to be before cloud_bucket cleas bucket.

        from google.cloud import storage
        if bucket not in self.storage_clients:
            cloud_bucket = CloudyGo.SECRET_CLOUD_BUCKET
            if cloud_bucket != CloudyGo.FULL_GAME_CLOUD_BUCKET:
                # the secret bucket name includes part of the bucket name.
                cloud_bucket += bucket.split('x')[0]

            client = storage.Client(project="minigo-pub").bucket(cloud_bucket)
            self.storage_clients[bucket] = client

        # MINIGO-HACK
        if bucket in CloudyGo.MINIGO_TS:
            # Take a guess at based on timestamp
            hour_guess = CloudyGo.guess_hour_dir(filename)
            model = hour_guess

            path = os.path.join('sgf', 'full', hour_guess, filename)
            if bucket == CloudyGo.FULL_GAME_CLOUD_BUCKET:
                # MINIGO_PUB has an outer folder of the bucket name
                path = os.path.join(bucket, path)
        else:
            path = os.path.join(bucket, 'sgf', model, 'full', filename)

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
        assert 1520000000 < file_time < 1550000000, file_time
        dt = datetime.utcfromtimestamp(file_time)
        return dt.strftime("%Y-%m-%d-%H")

    def get_game_data(self, bucket, model, filename, view_type):
        # Reconstruct path from filename

        base_path = os.path.join(self.sgf_path(bucket), model)
        if view_type == 'eval':
            base_path = os.path.join(self.data_path(bucket))

        file_path = os.path.join(base_path, view_type, filename)

        # MINIGO-HACK
        if (view_type != 'eval' and
                bucket in CloudyGo.MINIGO_TS and
                not os.path.isfile(file_path)):
            # Take a guess at based on timestamp
            hour_guess = CloudyGo.guess_hour_dir(filename)
            base_path = os.path.join(self.sgf_path(bucket), view_type)
            file_path = os.path.join(base_path, hour_guess, filename)

        base_dir_abs = os.path.abspath(base_path)
        file_path_abs = os.path.abspath(file_path)
        if not file_path_abs.startswith(base_dir_abs) or \
           not file_path_abs.endswith('.sgf'):
            return 'being naughty?', view_type

        data = ''
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                data = f.read()
        else:
            if 'full' in view_type \
                    and CloudyGo.LEELA_ID not in bucket \
                    and CloudyGo.FULL_GAME_CLOUD_BUCKET:
                data = self.__get_gs_game(bucket, model, filename, view_type)
                if data:
                    return data, view_type

            # Full games get deleted after X time, fallback to clean
            if 'full' in view_type:
                new_type = view_type.replace('full', 'clean')
                return self.get_game_data(bucket, model, filename, new_type)

        return data, view_type

    def _get_existing_games(self, where, args):
        query = ('SELECT timestamp, game_num, has_stats '
                 'FROM games '
                 'WHERE ' + where)
        return {(ts,g_n): h_s for ts, g_n, h_s in self.query_db(query, args)}

    def _get_games_from_model(self, model_id):
        return self._get_existing_games('model_id = ?', model_id)

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
            if model_id == 80:
                print (model_id, name, group, len(sgf), value)

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
            filename=None, max_nodes=None):
        # Old style
        # data = self.query_db(
        #    'SELECT cord, policy, n '
        #    'FROM position_eval_part '
        #    'WHERE model_id = ? AND type = ? AND name = ? '
        #    'ORDER BY n DESC, policy DESC',
        #    (model_id, group, name))

        board_size = CloudyGo.bucket_to_board_size(bucket)
        if max_nodes is None:
            max_nodes = board_size + 1

        # The V stands for 'value' as in 'policy value'
        # I'm very sorry about that future reader.
        high_v = sorted([p for c, p, n in data], reverse=True)
        cutoff = high_v[max_nodes] if len(high_v) > max_nodes else 0.01

        position_nodes = []
        for q, (cord, policy, n) in enumerate(data, 1):
            if 0 <= cord < board_size*board_size and (n > 0 or policy > cutoff):
                j, i = divmod(cord, board_size)
                value = q if n > 0 else round(100 * policy, 1)

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

        return [(move, round(100 * count / num_games))
                for move, count in favorite_openings
                if move and num_games and 100 * count >= num_games]

    #### PAGES ####

    def update_models(self, bucket, only_create=False):
        # LEELA-HACK
        if CloudyGo.LEELA_ID in bucket:
            model_glob = os.path.join(self.model_path(bucket), '[0-9a-f]*')
        else:
            model_glob = os.path.join(self.model_path(bucket), '*.meta')
        model_filenames = glob.glob(model_glob)

        existing_models = self.get_models(bucket)
        existing = set(m[0] for m in existing_models)

        model_inserts = []
        model_stat_inserts = []
        for model_filename in sorted(model_filenames):
            raw_name = os.path.basename(model_filename).replace('.meta', '')

            # LEELA-HACK
            if CloudyGo.LEELA_ID in bucket:
                # Note: this is brittle but I can't think of how to get model_id
                lz = [m for m in existing_models
                            if raw_name.startswith(m[1])]
                assert len(lz) == 1, (model_filename, lz)
                lz = lz[0]

                raw_name = raw_name[:8]  # otherwise sgf loading fails
                model_id = lz[0]
                model_name = lz[1]
                model_num = lz[4]
            else:
                model_num, model_name = raw_name.split('-', 1)
                model_id = CloudyGo.bucket_salt(bucket) + int(model_num)

            if only_create and model_id in existing:
                continue

            last_updated = int(time.time())
            training_time_m = 120
            creation = int(os.path.getmtime(model_filename))

            if last_updated - creation > (CloudyGo.FAST_UPDATE_HOURS * 3600):
                continue

            num_games = self.query_db(
                'SELECT count(*), sum(has_stats) from games WHERE model_id = ?',
                (model_id,))
            assert len(num_games) == 1
            num_games, num_stats_games = num_games[0]
            num_stats_games = num_stats_games or 0

            num_eval_games = self.query_db(
                'SELECT sum(games) from eval_models '
                'WHERE model_id_1 = ? AND model_id_2 = 0',
                (model_id,))
            num_eval_games = num_eval_games[0][0] or 0

            model = (
                model_id, model_name, raw_name,
                bucket, int(model_num),
                last_updated, creation, training_time_m,
                num_games, num_stats_games, num_eval_games
            )
            assert len(model) == 11, model
            model_inserts.append(model)

            currently_processed = self.query_db(
                'SELECT max(stats_games) FROM model_stats WHERE model_id = ?',
                (model_id,))
            currently_processed = currently_processed[0][0] or 0
            if num_games == currently_processed:
                continue

            opening_name = str(model_id) + '-favorite-openings.png'
            opening_file = os.path.join(
                self.INSTANCE_PATH, 'openings', opening_name)
            opening_sgf = sgf_utils.board_png(
                CloudyGo.bucket_to_board_size(bucket),
                '',  # setup
                self.get_favorite_openings(model_id, num_games),
                opening_file,
                force_refresh=True)

            # NOTE: if table is altered * may return unexpected order
            games = self.query_db(
                'SELECT * from games WHERE model_id = ?',
                (model_id,))

            for perspective in ['all', 'black', 'white']:
                is_all = perspective == 'all'
                is_black = perspective == 'black'

                # ASSUMPTION: every game has a result
                wins = games
                if not is_all:
                    wins = [game for game in games if game[4] == is_black]

                wins_by_resign = len([1 for game in wins if '+R' in game[5]])
                sum_wins_result = sum(float(game[5][2:]) for game in wins
                                      if '+R' not in game[5])

                resign_rates = Counter(game[17] for game in wins)
                if len(resign_rates) > 2 and CloudyGo.LEELA_ID not in bucket:
                    if perspective == 'all':
                        print('{} has multiple Resign rates: {}'.format(
                            raw_name, resign_rates))
                resign_rates.pop(-1, None)  # remove -1 if it's present

                # Note resign_rates are negative, max gets the value closest to zero.
                resign_rate = max(resign_rates.keys()) if resign_rates else -1
                assert resign_rate < 0, resign_rate

                # TODO count leela holdouts but ignore resign problem.
                holdouts = [game for game in wins if abs(game[17]) == 1]
                holdout_resigns = [
                    game for game in holdouts if '+R' in game[5]]
                assert len(holdout_resigns) == 0, holdout_resigns

                bad_resigns = 0
                for game in holdouts:
                    black_won = game[4]

                    # bleakest eval is generally negative for black and positive for white
                    black_would_resign = game[18] < resign_rate
                    white_would_resign = -game[19] < resign_rate

                    if black_won:
                        if black_would_resign:
                            bad_resigns += 1
                    else:
                        if white_would_resign:
                            bad_resigns += 1

                model_stat_inserts.append((
                    model_id, perspective,
                    num_games, num_stats_games,
                    len(wins), wins_by_resign,
                    len(wins) - wins_by_resign, sum_wins_result,
                    len(holdouts), bad_resigns,

                    sum(game[7] for game in wins),  # num_moves
                    sum(game[11] + game[12]
                        for game in wins),  # both sides visits
                    sum(game[13] + game[14]
                        for game in wins),  # both sides early visits
                    sum(game[15] + game[16]
                        for game in wins),  # both sides unluckiness
                    opening_sgf if is_all else '',  # favorite_openings
                ))

        db = self.db()

        cur = db.executemany(
            'DELETE FROM models WHERE model_id = ?',
            set((model[0],) for model in model_inserts))
        removed = cur.rowcount

        cur = db.executemany(
            'DELETE FROM model_stats WHERE model_id = ?',
            set((model[0],) for model in model_stat_inserts))
        removed_stats = cur.rowcount

        print('updated {}:  {} existing, {}|{} removed, {}|{} inserts'.format(
            bucket,
            len(existing),
            removed, removed_stats,
            len(model_inserts), len(model_stat_inserts)))

        self.insert_rows_db('models', model_inserts)
        self.insert_rows_db('model_stats', model_stat_inserts)

        db.commit()

        return len(model_inserts) + len(model_stat_inserts)

    @staticmethod
    def process_game(data):
        game_path, game_num, filename, model_id = data
        sgf_model, *result = sgf_utils.parse_game(game_path)
        if not result:
            return None

        # MINIGO-HACK
        is_ts_game = any(b in game_path for b in CloudyGo.MINIGO_TS)
        if sgf_model.isdigit() and is_ts_game:
            model_id = int(sgf_model)

        return game_num + (model_id, filename) + tuple(result)

    def update_model_names(self):
        model_names = dict(self.query_db('SELECT model_id, raw_name from models'))
        query = self.query_db('SELECT model_id, name FROM name_to_model_id')

        names = defaultdict(set)
        for m, n in query:
            names[m].add(n)

        inserts = []
        for model_id, model_name in sorted(model_names.items()):
            if model_name not in names[model_id]:
                inserts.append((model_name, model_id))
                names[model_id].add(model_name)

            # LEELA-HACK: also add short name.
            if re.match(r'$[0-9a-fA-F]{64}^', model_name):
                short_name = model_name[:8]
                if short_name not in names[model_id]:
                    inserts.append((short_name, model_id))
                    names[model_id].add(short_name)

        # MINIGO-HACK
        for model_id, name in sorted(model_names.items()):
            bucket = [b for b in CloudyGo.MINIGO_TS
                      if model_id in range(*CloudyGo.bucket_model_range(b))]
            if not bucket:
                continue
            bucket = bucket[0]

            cur = any(n.startswith(CloudyGo.MODEL_CKPT)
                      for n in names[model_id])
            if len(name) > 3 and not cur:
                # VERY SLOW
                print ("Slow lookup of checkpoint step for", model_id, name)
                path = os.path.join(self.model_path(bucket), name)
                import tensorflow.train as tf_train
                ckpt = tf_train.load_checkpoint(path)
                step = ckpt.get_tensor('global_step')
                new_name = CloudyGo.MODEL_CKPT + str(step)
                assert new_name not in names[model_id], (new_name, model_id)
                names[model_id].add(new_name)

                inserts.append((new_name, model_id))
                print ("\t", model_id, name, " =>  ", new_name)

        if inserts:
            self.insert_rows_db('name_to_model_id', inserts)
            self.db().commit()
            print('Updated {} model names'.format(len(inserts)))

    def get_model_names(self, model_range):
        raw_names = self.query_db(
            'SELECT model_id, name FROM name_to_model_id '
            'WHERE model_id BETWEEN ? AND ?',
            model_range)

        num_to_name = {}
        for model_id, name in raw_names:
            if name.startswith(CloudyGo.MODEL_CKPT):
                continue

            if "ELF" in name:
                continue

            if model_id in num_to_name:
                # NOTE(sethtroisi): Models having multiple names complicates
                # things, e.g. eval page.
                print("ERROR: {}, multiple names: {}, {}".format(
                    model_id, name, num_to_name[model_id]))

            num_to_name[model_id] = name

        for model_id, name in raw_names:
            if model_id in num_to_name:
                continue

            # if no name other than model.ckpt, use that name
            if name.startswith(CloudyGo.MODEL_CKPT):
                if model_id % CloudyGo.SALT_MULT < CloudyGo.CROSS_EVAL_START:
                    print("error: {} only had ckpt name ({})".format(
                        model_id, name))
                num_to_name[model_id] = name

        return num_to_name

    def update_bucket_ranges(self, bucket_names):
        buckets = self.query_db('SELECT bucket from bucket_model_range')
        buckets = set([b[0] for b in buckets])

        for bucket in bucket_names:
            if bucket not in buckets:
                model_range = CloudyGo.bucket_model_range(bucket)
                print("Adding {} => {}".format(bucket, model_range))
                self.insert_rows_db(
                    'bucket_model_range',
                    [(bucket,) + model_range])
                self.db().commit()

    @staticmethod
    def _model_guesser(filename, model_mtimes, model_ids):
        game_time = int(filename.split('-', 1)[0])
        game_time -= CloudyGo.MINIGO_GAME_LENGTH
        assert 1520000000 < game_time < 1550000000, game_time
        model_num = bisect.bisect(model_mtimes, game_time, 1) - 1
        assert model_num >= 0, model_num
        return model_ids[model_num]

    @staticmethod
    def _game_paths_to_to_process(
            bucket, existing,
            model_lookup,
            game_paths,
            max_inserts):

        bucket_salt = CloudyGo.bucket_salt(bucket)
        to_process = []

        for game_path in sorted(game_paths):
            filename = os.path.basename(game_path)
            game_num = CloudyGo.get_game_num(bucket_salt, filename)
            has_stats = '/full/' in game_path

            # Skip if already processed UNLESS was processed without stats.
            current = existing.get(game_num, None)
            if current or (current == False and not has_stats):
                continue

            existing[game_num] = has_stats

            # MINIGO-HACK, this would be easy otherwise
            model_id = model_lookup(filename)
            to_process.append((game_path, game_num, filename, model_id))

            if len(to_process) >= max_inserts:
                break
        return to_process


    def _get_update_games_time_dir(self, bucket, max_inserts):
        model_range = CloudyGo.bucket_model_range(bucket)
        models = self.get_models(bucket)

        model_mtimes = [model[6] for model in models]
        model_ids = [model[0] for model in models]
        model_lookup = functools.partial(
            CloudyGo._model_guesser,
            model_mtimes=model_mtimes,
            model_ids=model_ids)

        # NOTE: this code goes first so we know what timestamp range to load.
        to_update = OrderedDict()
        for d_type in ['full', 'clean']:
            base_paths = os.path.join(self.sgf_path(bucket), d_type, '*')
            time_dirs = sorted(glob.glob(base_paths))
            # FAST UPDATE HACK
            will_update = time_dirs[-CloudyGo.FAST_UPDATE_HOURS:]
            if len(will_update) > 0:
                to_update[d_type] = will_update
            print ("\t{}, {} folders (updating {}): {}".format(
                d_type,
                len(time_dirs),
                len(will_update),
                utils.list_preview(list(map(os.path.basename, time_dirs)), 1)))

        if len(to_update) == 0:
            return

        def get_folder_ts(func):
            all_updates = [d for v in to_update.values() for d in v]
            folder = func(map(os.path.basename, all_updates))
            dt = datetime.strptime(folder, '%Y-%m-%d-%H')
            return int(dt.replace(tzinfo=timezone.utc).timestamp())

        min_ts = get_folder_ts(min) - 5
        max_ts = get_folder_ts(max) + 3605

        existing = self._get_games_from_ts(model_range, (min_ts, max_ts))
        print("{} existing games ({} to {})".format(
            len(existing), min_ts, max_ts))

        # TODO find a way to rsync faster
        for d_type, time_dirs in to_update.items():
            for time_dir in time_dirs:
                name = os.path.basename(time_dir)

                game_paths = glob.glob(os.path.join(time_dir, '*.sgf'))
                to_process = CloudyGo._game_paths_to_to_process(
                    bucket, existing, model_lookup, game_paths, max_inserts)
                yield name, to_process, len(existing)


    def _get_update_games_model(self, bucket, max_inserts):
        skipped = []
        for model in self.get_models(bucket):
            # FAST UPDATE HACK: only newest day of model folders are updated.
            # Check if directory mtime is recent, if not skip
            test_d = os.path.join(self.sgf_path(bucket), model[2], 'full')
            m_time = os.path.getmtime(test_d) if os.path.exists(test_d) else 0
            if model[5] > m_time + 3600 * CloudyGo.FAST_UPDATE_HOURS:
                skipped.append(model[4])
                continue

            name = '{}-{}'.format(model[4], model[1])
            model_id = model[0]
            model_lookup = lambda: model_id

            existing = self._get_games_from_model(model_id)
            # TODO: Support update from clean if so desired.
            game_paths = self.all_games(bucket, model[2])
            to_process = CloudyGo._game_paths_to_to_process(
                bucket, existing, model_lookup, game_paths, max_inserts)

            yield name, to_process, len(existing)
        if len(skipped) > 0:
            print('skipped {}, {}'.format(
                len(skipped), utils.list_preview(skipped)))


    def map_and_filter(self, funct, items, unit="it"):
        length = len(items)
        mapper = self.pool.imap if self.pool else itertools.imap
        mapped = mapper(funct, items)

        if length < 100:
            results = list(mapped)
        else:
            results = list(tqdm(mapped, unit=unit, total=length))

        broken = results.count(None)
        if broken > 10:
            print("{} Broken".format(broken))

        return list(filter(None.__ne__, results))


    def update_games(self, bucket, max_inserts):
        # This is REALLY SLOW because it's potentially >1M items
        # loop by model to avoid huge globs and commits
        updates = 0
        bucket_salt = CloudyGo.bucket_salt(bucket)

        if bucket in CloudyGo.MINIGO_TS:
            games_source = self._get_update_games_time_dir(bucket, max_inserts)
        else:
            games_source = self._get_update_games_model(bucket, max_inserts)

        for model_name, to_process, len_existing in games_source:
            if len(to_process) > 0:
                print("About to process {} games of {}".format(
                    len(to_process), model_name))

                new_games = self.map_and_filter(CloudyGo.process_game, to_process)

                # Some Games were processed as clean may now have stats data.
                self.insert_rows_db('games', new_games, allow_existing=True)
                self.db().commit()

                result = '{}: {} existing, {} inserts'.format(
                    model_name, len_existing, len(new_games))

                if len(new_games):
                    print(result)
                updates += len(new_games)
        return updates

    def update_position_eval(self, filename, bucket, model_id, group, name):
        board_size = CloudyGo.bucket_to_board_size(bucket)

        with open(filename) as eval_file:
            values = list(map(str.strip, eval_file.read().strip().split(',')))

            # Files start with model_id (idx)
            test = values.pop(0)
            assert test in str(model_id), \
                '{} not in {}'.format(test, model_id)

            value = values.pop(0)
            policy = None
            n = None
            sgf = None

            if group == 'policy':
                assert len(values) in (9*9+1, 19*19+1), (group, filename)

                render = (name == 'empty' and group == 'policy')
                filename = '-'.join([str(model_id), group, name + '.png'])
                filepath = os.path.join(
                    self.INSTANCE_PATH, 'openings', filename)

                data = [(cord, float(policy), 0)
                        for cord, policy in enumerate(values)]

                sgf = self.render_position_eval(
                    bucket, model_id,
                    group, name, data,
                    filename=filepath if render else None)
            else:
                assert group == 'pv', '{} {} {}'.format(
                    group, len(values), filename)
                assert len(values) % 2 == 0, values

                cords = set()
                data = []
                for i in range(0, len(values), 2):
                    cord = int(values[i])
                    count = int(values[i+1])
                    if cord in cords:
                        # TODO remove this constraint
                        # Playing at a previous location (AKA fighting a ko)
                        continue
                    cords.add(cord)
                    data.append((cord, 0, count))

                sgf = self.render_position_eval(
                    bucket, model_id, group, name, data)

            sgf = sgf_utils.canonical_sgf(board_size, sgf)
            insert = (model_id, -2, group, name, policy, value, n, sgf)

            self.db().execute(
                'DELETE FROM position_eval_part '
                'WHERE model_id = ? AND type = ? AND name = ?',
                (model_id, group, name))

            self.insert_rows_db('position_eval_part', [insert])
            self.db().commit()

    @staticmethod
    def process_eval(data):
        eval_path, filename, eval_num, model_id_1, model_id_2 = data
        result = sgf_utils.parse_game_simple(eval_path, include_players=True)
        if not result:
            return None
        return (eval_num, filename, model_id_1, model_id_2) + result

    @staticmethod
    def sanitize_player_name(name):
        # See oneoff/leela-all-to-dirs.sh
        name = re.sub(
            r'Leela\s*Zero\s*([0-9](\.[0-9]+)*)?\s+(networks)?\s*', '', name)
        name = re.sub(r'([0-9a-f]{8})[0-9a-f]{56}', r'\1', name)
        return name

    def process_sgf_names(self, bucket, records):
        model_range = CloudyGo.bucket_model_range(bucket)
        bucket_salt = model_range[0]
        name_to_num = dict(self.query_db(
            'SELECT name, model_id FROM name_to_model_id '
            'WHERE model_id BETWEEN ? AND ?',
            model_range))
        new_names = []

        def ckpt_num(name):
            if name.startswith(CloudyGo.MODEL_CKPT):
                return int(name[len(CloudyGo.MODEL_CKPT):])
            return None

        def get_or_add_name(name):
            name = CloudyGo.sanitize_player_name(name)

            if name in name_to_num:
                return name_to_num[name]

            # HACK: figure out how to plumb is_sorted here
            if (bucket.startswith('v') and
                re.match(r'[0-9]{6}-([a-zA-Z-]+)', name)):
                return bucket_salt + int(name.split('-', 1)[0])

            # MINIGO-HACK: bucket ~= 'v10-19x19', name ~= 'model.ckpt.123'
            if bucket.startswith('v') and ckpt_num(name):
                num = ckpt_num(name)
                previous = set(ckpt_num(other) for other in name_to_num.keys())
                count_less = sum(1 for p in previous if p and p < num)
                # ckpt-0 and ckpt-1 both mean 000000-bootstrap.
                # awkwardly count_less filters ckpt-0 so the count is correct.
                number = bucket_salt + count_less
                name_to_num[name] = number
                new_names.append((name, number))
                print("get_or_add_name ckpt:", name, number)
                return number

            first_eval_model = bucket_salt + CloudyGo.CROSS_EVAL_START
            keys = set(name_to_num.values())
            for test_id in range(first_eval_model, model_range[1]):
                if test_id not in keys:
                    name_to_num[name] = test_id
                    new_names.append((name, test_id))
                    return test_id
            assert False

        new_records = []
        for record in records:
            # Eval records
            if bucket_salt == record[2] == record[3]:
                new_record = list(record[:-2])
                PB, PW = record[-2:]
                new_record[2] = get_or_add_name(PB)
                new_record[3] = get_or_add_name(PW)
            elif ckpt_num(record[1]) is not None:
                # MINIGO-HACK
                new_record = list(record)
                model = record[1]
                new_record[1] = get_or_add_name(model)
            else:
                # Eval game that has well formed model_ids.
                # TODO plumb is_sorted around and don't set model_ids
                new_record = list(record[:-2])

            assert new_record, record
            new_records.append(tuple(new_record))

        # NOTE: this depends on someone else calling db.commit()
        if new_names:
            self.insert_rows_db('name_to_model_id', new_names)

        return new_records

    def update_eval_games(self, bucket):
        eval_dir = self.eval_path(bucket)
        if not os.path.exists(eval_dir):
            return 0

        bucket_salt = CloudyGo.bucket_salt(bucket)

        existing = self._get_eval_games(bucket)
        evals_to_process = []
        new_eval_nums = set()

        eval_games = glob.glob(
            os.path.join(eval_dir, '**', '*.sgf'),
            recursive=True)

        # sort by newest first
        eval_games = sorted(eval_games, reverse=True)
        for eval_path in eval_games:
            # NOTE: we want to keep the YYYY-MM-DD folder part
            partial_path = eval_path.replace(eval_dir + '/', '')
            filename = os.path.basename(eval_path)

            # NOTE: for files from ringmaster we use partial_path which
            # is generally the ctl filename.
            if re.match('[0-9]+_[0-9]+\.sgf', filename):
                filename = partial_path

            eval_num, m1, m2 = CloudyGo.get_eval_parts(filename)
            if eval_num in existing:
                continue
            assert eval_num not in new_eval_nums, (eval_num, filename)
            new_eval_nums.add(eval_num)

            # Minigo eval games have white before black
            white_model = bucket_salt + m1
            black_model = bucket_salt + m2

            evals_to_process.append(
                (eval_path,
                 partial_path,
                 eval_num,
                 black_model,
                 white_model))

            if len(evals_to_process) >= CloudyGo.MAX_INSERTS:
                break

        new_evals = []
        if len(evals_to_process) > 0:
            print()
            new_evals = self.map_and_filter(
                CloudyGo.process_eval,
                evals_to_process,
                unit="eval games")

            # Post-process PB/PW to model_id (when filename is ambiguous)
            new_evals = self.process_sgf_names(bucket, new_evals)

            if new_evals:
                self.insert_rows_db('eval_games', new_evals)
                self.db().commit()

            print('eval_games: {} existing, {} inserts'.format(
                len(existing), len(new_evals)))
        return len(new_evals)

    def update_eval_models(self, bucket):
        model_range = CloudyGo.bucket_model_range(bucket)

        eval_games = self.query_db(
            'SELECT '
            '   model_id_1, '
            '   model_id_2, '
            '   black_won '
            'FROM eval_games '
            'WHERE model_id_1 BETWEEN ? AND ?',
            model_range)

        if len(eval_games) < 10:
            return 0

        total_games = len(eval_games)
        model_nums = sorted(
            set(e[0] for e in eval_games) |
            set(e[1] for e in eval_games)
        )

        print('loaded {} evals for {} models ({} to {})'.format(
            total_games, len(model_nums),
            min(model_nums, default=-1),
            max(model_nums, default=-1)))

        ratings = CloudyGo.get_eval_ratings(model_nums, eval_games)
        assert 0 not in ratings, ratings.keys()

        # black games, black wins,    white games, white wins
        model_evals = defaultdict(lambda: [0, 0, 0, 0])

        def increment_record(record, played_black, black_won):
            record[0 if played_black else 2] += 1
            if played_black == black_won:
                record[1 if played_black else 3] += 1

        for d in eval_games:
            black, white, black_won = d

            # Update by model
            increment_record(model_evals[(white, 0)], False, black_won)
            increment_record(model_evals[(black, 0)], True, black_won)

            # Update by pairing
            increment_record(model_evals[(white, black)], False, black_won)
            increment_record(model_evals[(black, white)], True, black_won)

        records = []
        for (m1, m2), (m1_b, m1_b_wins, m1_w, m1_w_wins) in model_evals.items():
            m1_rating = ratings[m1]
            m2_rating = ratings[m2] if m2 != 0 else m1_rating
            records.append([
                m1, m2,
                (m1_rating[0] + m2_rating[0]) / 2,
                # Note V std err should be combine (sum std_err^2)^(1/2)
                (m1_rating[1] + m2_rating[1]) / 2,

                m1_b + m1_w,
                m1_b, m1_b_wins,
                m1_w, m1_w_wins,
            ])

        # Delete old eval_models and add new eval games
        cur = self.db().execute(
            'DELETE FROM eval_models WHERE model_id_1 BETWEEN ? AND ?',
            model_range)
        removed = cur.rowcount

        self.insert_rows_db('eval_models', records)
        self.db().commit()

        self.db().executemany(
            'UPDATE models SET num_eval_games = ? WHERE model_id = ?',
            [(k[0], d[0] + d[2]) for k, d in model_evals.items() if k[1] == 0])
        self.db().commit()

        return len(records) - removed

    @staticmethod
    def get_eval_ratings(model_nums, eval_games):
        # Map model_nums to a contigious range.
        ordered = sorted(set(model_nums))
        new_num = {}
        for i, m in enumerate(ordered):
            new_num[m] = i

        def ilsr_data(eval_game):
            p1, p2, black_won = eval_game
            p1 = new_num[p1]
            p2 = new_num[p2]
            assert 0 <= p1 <= 3500
            assert 0 <= p2 <= 3500

            return (p1, p2) if black_won else (p2, p1)

        pairs = list(map(ilsr_data, eval_games))
        ilsr_param = choix.ilsr_pairwise(
            len(ordered),
            pairs,
            alpha=0.0001,
            max_iter=200)

        hessian = choix.opt.PairwiseFcts(pairs, penalty=.1).hessian(ilsr_param)
        std_err = np.sqrt(np.diagonal(np.linalg.inv(hessian)))

        # Elo conversion
        elo_mult = 400 / math.log(10)

        min_rating = min(ilsr_param)
        ratings = {}
        for model, param, err in zip(ordered, ilsr_param, std_err):
            ratings[model] = (elo_mult * (param - min_rating), elo_mult * err)

        return ratings
