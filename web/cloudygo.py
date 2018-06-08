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

import datetime
import glob
import hashlib
import math
import os
import re
import time
import zlib
from collections import Counter, defaultdict

import choix
import numpy as np

from . import sgf_utils
from . import utils


class CloudyGo:
    SALT_MULT = 10 ** 6

    GAME_TIME_MOD = 10 ** 8 # Most of a year in seconds
    GAME_POD_MULT = 10 ** 8 # Large enough for 6 hexdecimal characters
    GAME_BUCKET_MULT = 10 ** 3

    DIR_EVAL_START = 2500 # offset from SALT_MULT to start

    # set by __init__ but treated as constant
    INSTANCE_PATH = None
    DATA_DIR = None

    # set to 'minigo-pub' or similiar to serve debug games from the cloud.
    DEBUG_GAME_CLOUD_BUCKET = 'minigo-pub'

    DEFAULT_BUCKET = 'v7-19x19'
    LEELA_ID = 'leela-zero'

    def __init__(self, instance_path, data_dir, database, cache, pool):
        self.INSTANCE_PATH = instance_path
        self.DATA_DIR = data_dir

        self.db = database
        self.cache = cache
        self.pool = pool

        self.last_cloud_request = 0
        self.storage_client = None

    #### PATH UTILS ####


    def data_path(self, bucket):
        return '{}/{}'.format(self.DATA_DIR, bucket)

    def model_path(self, bucket):
        return os.path.join(self.data_path(bucket), 'models')

    def sgf_path(self, bucket):
        return os.path.join(self.data_path(bucket), 'sgf')

    def eval_path(self, bucket):
        return os.path.join(self.data_path(bucket), 'eval')


    def all_games(self, bucket, model, game_type='full'):
        # LEELA-HACK
        if CloudyGo.LEELA_ID in bucket:
            game_type = 'clean'

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
        return ' model_id >= {} AND model_id < {} '.format(
            min_model, min_model + CloudyGo.SALT_MULT)


    def bucket_query_db(
            self, bucket,
            select, table, where,
            group_count, limit = 1000, args=()):
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


    def insert_rows_db(self, table, rows):
        assert re.match('^[a-zA-Z_2]*$', table), table
        if len(rows) > 0:
            values = '({})'.format(','.join(['?'] * len(rows[0])))
            query = 'INSERT INTO {} VALUES {}'.format(re.escape(table), values)
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
        return (bucket_salt, bucket_salt + CloudyGo.SALT_MULT)


    @staticmethod
    def get_game_num(bucket_salt, filename):
        # LEELA-HACK
        if CloudyGo.LEELA_ID in filename:
            number = filename.rsplit('-', 1)[-1]
            assert number.endswith('.sgf')
            # TODO(sethtroisi): these are generated from sgfsplit
            # come up with a better scheme (hash maybe) for numbering
            return int(number[:-4])

        game, name = filename.split('-', 1)
        pod = name.split('-')[-1][:-4]

        game_num = int(game) % CloudyGo.GAME_TIME_MOD

        pod_num = int(pod, 36)
        assert pod_num < CloudyGo.GAME_POD_MULT
        game_num = game_num * CloudyGo.GAME_POD_MULT + pod_num

        assert bucket_salt % CloudyGo.SALT_MULT == 0, bucket_salt
        bucket_num = bucket_salt // CloudyGo.SALT_MULT
        game_num = game_num * CloudyGo.GAME_BUCKET_MULT + bucket_num

        return game_num


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
        now = datetime.datetime.now()
        was = datetime.datetime.fromtimestamp(mtime)
        delta = now - was
        deltaDays = str(delta.days) + ' days '
        deltaHours = str(round(delta.seconds / 3600, 1)) + ' hours ago'

        return [
            was.date().isoformat(),
            (deltaDays if delta.days > 0 else '') + deltaHours
        ]


    def get_models(self, bucket):
        return self.query_db(
            'SELECT * FROM models WHERE bucket = ?',
            (bucket,))


    def get_newest_model_num(self, bucket):
        model_nums = self.query_db(
            'SELECT num FROM models WHERE bucket = ? ORDER BY num DESC LIMIT 1',
            (bucket,))
        assert len(model_nums) > 0, model_nums
        return model_nums[0][0]


    def load_model(self, bucket, model_name):
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
                'SELECT * FROM games2 WHERE game_num = ?',
                (game_num,))
            if len(game) == 0:
                continue
            games.append(game[0])
        return games


    def __get_gs_game(self, bucket, model, filename, view_type):
        assert 'full' in view_type, view_type

        # Maybe it's worth caching these for, now just globally rate limit
        now = time.time()
        if now - self.last_cloud_request < 1:
            return None
        self.last_cloud_request = now

        from google.cloud import storage
        if not self.storage_client:
            self.storage_client = storage.Client().bucket(
                CloudyGo.DEBUG_GAME_CLOUD_BUCKET)

        path = os.path.join(bucket, 'sgf', model, 'full', filename)
        blob = self.storage_client.get_blob(path)
        if not isinstance(blob, storage.Blob):
            return None

        data = blob.download_as_string().decode('utf8')
        return data


    def get_game_data(self, bucket, model, filename, view_type):
        # Reconstruct path from filename

        base_path = os.path.join(self.sgf_path(bucket), model)
        if view_type == 'eval':
            base_path = os.path.join(self.data_path(bucket))

        file_path = os.path.join(base_path, view_type, filename)

        base_dir_abs = os.path.abspath(base_path)
        file_path_abs = os.path.abspath(file_path)
        if not file_path_abs.startswith(base_dir_abs) or \
           not file_path_abs.endswith('.sgf'):
            return 'being naughty?'

        print (file_path)

        data = ''
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                data = f.read()
        else:
            if 'full' in view_type \
                   and CloudyGo.LEELA_ID not in bucket \
                   and CloudyGo.DEBUG_GAME_CLOUD_BUCKET:
                data = self.__get_gs_game(bucket, model, filename, view_type)
                if data:
                    return data, view_type

            # Full games get deleted after X time, fallback to clean
            if 'full' in view_type:
                new_type = view_type.replace('full', 'clean')
                return self.get_game_data(bucket, model, filename, new_type)

        return data, view_type


    def get_existing_games(self, model_id):
        query = 'SELECT game_num FROM games2 WHERE model_id = ?'
        return set(record[0] for record in self.query_db(query, (model_id,)))


    def get_existing_eval_games(self, bucket):
        model_range = CloudyGo.bucket_model_range(bucket)
        query = ('SELECT eval_num '
                'FROM eval_games '
                'WHERE model_id_1 >= ? and model_id_1 < ?')
        records = self.query_db(query, model_range)
        return set(r[0] for r in records)


    def get_position_sgfs(self, bucket, model_ids=None):
        bucket_salt = CloudyGo.bucket_salt(bucket)

        # Single model_id, two model_ids, all model_ids
        where = 'WHERE cord = -2 AND model_id >= ? AND model_id < ?'
        args  = (bucket_salt + 10, bucket_salt + CloudyGo.SALT_MULT)

        if model_ids == None:
            pass
        elif len(model_ids) == 1:
            args  = (model_ids[0], model_ids[0] + 1)
        elif len(model_ids) == 2:
            where = 'WHERE cord = -2 AND (model_id == ? or model_id == ?)'
            args  = (model_ids[0], model_ids[1])
        else:
            assert False, model_ids

        sgfs = self.query_db(
            'SELECT model_id, name, type, sgf, round(value,3) '
            'FROM position_eval_part ' + \
            where,
            args)

        arranged = defaultdict(lambda:defaultdict(lambda: defaultdict(lambda:('', 0))))
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

        names = sorted(names, key=lambda n:(setups.get(n, 10), name))

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
                j,i = divmod(cord, board_size)
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
            'FROM games2 WHERE model_id = ? '
            'GROUP BY 1 ORDER BY 2 DESC LIMIT 16',
            (model_id,))

        return [(move, round(100 * count / num_games))
            for move, count in favorite_openings
                if move and num_games and 100 * count >= num_games]


    #### PAGES ####


    def update_models(self, bucket, partial=True):
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
                existing_model = [m for m in existing_models
                    if raw_name.startswith(m[1])]
                assert len(existing_model) == 1, (model_filename, existing_model)
                existing_model = existing_model[0]

                raw_name = raw_name[:8] # otherwise sgf loading fails
                model_id = existing_model[0]
                model_name = existing_model[1]
                model_num = existing_model[4]
            else:
                model_num, model_name = raw_name.split('-', 1)
                model_id = CloudyGo.bucket_salt(bucket) + int(model_num) # unique_id

            last_updated = int(time.time())
            training_time_m = 120
            creation = int(os.path.getmtime(model_filename))

            num_games = self.query_db(
                'SELECT count(*), sum(has_stats) from games2 WHERE model_id = ?',
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

            if partial and model_id in existing:
                continue

            currently_processed = self.query_db(
                'SELECT max(stats_games) FROM model_stats WHERE model_id = ?',
                (model_id,))

            currently_processed = currently_processed[0][0] or 0
            if partial and num_games == currently_processed:
                continue

            opening_name = str(model_id) + '-favorite-openings.png'
            opening_file = os.path.join(
                self.INSTANCE_PATH, 'openings', opening_name)
            opening_sgf  = sgf_utils.board_png(
                CloudyGo.bucket_to_board_size(bucket),
                '', #setup
                self.get_favorite_openings(model_id, num_games),
                opening_file,
                force_refresh=True)

            # NOTE: if table is altered * may return unexpected order
            games = self.query_db(
                'SELECT * from games2 WHERE model_id = ?',
                (model_id,))

            for perspective in ['all', 'black', 'white']:
                is_all = perspective == 'all'
                is_black = perspective == 'black'

                # ASSUMPTION: every game has a result
                wins = games
                if not is_all:
                    wins = [game for game in games if game[3] == is_black]

                wins_by_resign = len([1 for game in wins if '+R' in game[4]])
                sum_wins_result = sum(float(game[4][2:]) for game in wins
                    if '+R' not in game[4])

                resign_rates = Counter(game[16] for game in wins)
                resign_rates.pop(-1, None) # remove -1 if it's present
                if len(resign_rates) > 1 and CloudyGo.LEELA_ID not in bucket:
                    if perspective == 'all':
                        print('{} has multiple Resign rates: {}'.format(
                            raw_name, resign_rates))

                resign_rate = min(resign_rates.keys()) if resign_rates else -1
                assert resign_rate < 0, resign_rate
                # TODO count leela holdouts but ignore resign problem.
                holdouts = [game for game in wins if abs(game[16]) == 1]
                holdout_resigns = [game for game in holdouts if '+R' in game[4]]
                assert len(holdout_resigns) == 0, holdout_resigns

                bad_resigns = 0
                for game in holdouts:
                    black_won = game[3]

                    # bleakest eval is generally negative for black and positive for white
                    black_would_resign = game[17] < resign_rate
                    white_would_resign = -game[18] < resign_rate

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

                    sum(game[6] for game in wins), # num_moves
                    sum(game[10] + game[11] for game in wins), # both sides visits
                    sum(game[12] + game[13] for game in wins), # both sides early visits
                    sum(game[14] + game[15] for game in wins), # both sides unluckiness
                    opening_sgf if is_all else '', # favorite_openings
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
        result = sgf_utils.parse_game(game_path)
        if not result: return None
        return  (game_num, model_id, filename) + result


    def update_model_names(self):
        models = self.query_db('SELECT * from models')
        names = dict(self.query_db(
            'SELECT model_id, name FROM name_to_model_id'))

        inserts = []
        for model in models:
            model_id = model[0]
            model_name = model[1]
            if model_id in names:
                assert model_name == names[model_id], (model_id, model_name)
            else:
                inserts.append((model_name, model_id))

        if inserts:
            self.insert_rows_db('name_to_model_id', inserts)
            self.db().commit()
            print ('Updated {} model names'.format(len(inserts)))


    def update_games(self, bucket, max_inserts):
        # This is REALLY SLOW because it's potentially >1M items
        # loop by model to avoid huge globs and commits
        updates = 0

        bucket_salt = CloudyGo.bucket_salt(bucket)

        skipped = []
        for model in self.get_models(bucket):
            # Check if directories mtime is recent, if not skip
            base = os.path.join(self.sgf_path(bucket), model[2])
            times = [0]
            for test_d in [os.path.join(base, d) for d in ('clean', 'full')]:
                if os.path.exists(test_d):
                    times.append(os.path.getmtime(test_d))
            if max(times) < model[5] - 86400:
                skipped.append(model[4])
                continue

            model_id = model[0]
            existing = self.get_existing_games(model_id)

            games_added = set()
            to_process = []
            for game_path in self.all_games(bucket, model[2]):
                filename = os.path.basename(game_path)
                game_num = CloudyGo.get_game_num(bucket_salt, filename)

                if game_num in existing: continue

                assert game_num not in games_added, (game_num, game_path)
                games_added.add(game_num)

                to_process.append((game_path, game_num, filename, model_id))

                if len(to_process) >= max_inserts:
                    break

            mapper = self.pool.map if self.pool else map
            new_games = mapper(CloudyGo.process_game, to_process)

            new_games = list(filter(None.__ne__, new_games))
            total_games = len(existing) + len(new_games)

            self.insert_rows_db('games2', new_games)

            self.db().execute(
                'UPDATE models SET num_games = ? WHERE model_id = ?',
                (total_games, model_id,))

            self.db().commit()

            result = '{}-{}: {} existing, {} inserts'.format(
                model[4], model[1], len(existing), len(new_games))
            if len(new_games):
                print (result)

            updates += len(new_games)

        if len(skipped) > 0:
            print ('skipped {}, {}'.format(
                len(skipped), utils.list_preview(skipped)))
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

            if len(values) in (9*9+1, 19*19+1):
                assert group == 'policy', group

                render = (name == 'empty' and group == 'policy')
                filename = '-'.join([str(model_id), group, name + '.png'])
                filepath = os.path.join(self.INSTANCE_PATH, 'openings', filename)

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
                    data.append( (cord, 0, count) )

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
        if not result: return None
        return (eval_num, filename, model_id_1, model_id_2) + result


    @staticmethod
    def sanitize_player_name(name):
        # See oneoff/leela-all-to-dirs.sh
        name = re.sub(r'Leela\s*Zero\s*([0-9](\.[0-9]+)*)?\s+(networks)?\s*', '', name)
        name = re.sub(r'([0-9a-f]{8})[0-9a-f]{56}', r'\1', name)
        return name


    def process_eval_names(self, bucket, eval_records):
        model_range = CloudyGo.bucket_model_range(bucket)
        bucket_salt = model_range[0]
        name_to_num = dict(self.query_db(
            'SELECT name, model_id FROM name_to_model_id '
            'WHERE model_id >= ? AND model_id < ?',
             model_range))

        new_names = []
        def get_or_add_name(name):
            name = CloudyGo.sanitize_player_name(name)

            if name in name_to_num:
                return name_to_num[name]

            keys = set(name_to_num.values())
            first_eval_model = bucket_salt + CloudyGo.DIR_EVAL_START
            for test_id in range(first_eval_model, model_range[1]):
                if test_id not in keys:
                    name_to_num[name] = test_id
                    new_names.append((name, test_id))
                    return test_id
            assert False

        new_evals = []
        for eval_record in eval_records:
            record = list(eval_record[:-2])
            if bucket_salt == record[2] == record[3]:
                PB, PW = eval_record[-2:]
                record[2] = get_or_add_name(PB)
                record[3] = get_or_add_name(PW)

            new_evals.append(tuple(record))

        return new_evals, new_names


    def update_eval_games(self, bucket):
        eval_dir = self.eval_path(bucket)
        if not os.path.exists(eval_dir):
            return 0

        bucket_salt = CloudyGo.bucket_salt(bucket)

        existing = self.get_existing_eval_games(bucket)
        evals_to_process = []

        # TODO convince andrew to store these by YYYY-MM-DD, and walk dirs
        eval_games = glob.glob(os.path.join(eval_dir, '*.sgf'))

        # sort by newest first
        eval_games = sorted(eval_games, reverse=True)
        print ("Found {} eval games".format(len(eval_games)))
        for eval_path in eval_games:
            filename = os.path.basename(eval_path)

            eval_num, m1, m2 = CloudyGo.get_eval_parts(filename)
            if eval_num in existing: continue

            # Minigo eval games have white before black
            white_model = bucket_salt + m1
            black_model = bucket_salt + m2

            evals_to_process.append(
                (eval_path,
                 filename,
                 eval_num,
                 black_model,
                 white_model))

        new_evals = []
        if len(evals_to_process) > 0:
            mapper = self.pool.map if self.pool else map
            new_evals = mapper(CloudyGo.process_eval, evals_to_process)

            broken = new_evals.count(None)
            new_evals = list(filter(None.__ne__, new_evals))

            if broken > 10:
                print ("{} Broken games".format(broken))

            # Post-process PB/PW to model_id (when filename is ambiguous)
            new_evals, new_names = self.process_eval_names(bucket, new_evals)

            if new_names:
                self.insert_rows_db('name_to_model_id', new_names)

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
            'WHERE model_id_1 >= ? AND model_id_1 < ?',
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

        # black games, black wins,    white games, white wins
        model_evals = defaultdict(lambda : [0,0,0,0])

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

        ratings = CloudyGo.get_eval_ratings(model_nums, eval_games)
        assert 0 not in ratings, ratings.keys()

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
        self.db().execute(
            'DELETE FROM eval_models WHERE model_id_1 >= ? AND model_id_1 < ?',
            model_range)
        self.insert_rows_db('eval_models', records)
        self.db().commit()

        self.db().executemany(
            'UPDATE models SET num_eval_games = ? WHERE model_id = ?',
            [(k[0], d[0] + d[2]) for k, d in model_evals.items() if k[1] == 0])
        self.db().commit()

        return len(records)


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
            assert 0 <= p1 <= 3000
            assert 0 <= p2 <= 3000

            return (p1, p2) if black_won else (p2, p1)

        pairs = list(map(ilsr_data, eval_games))
        ilsr_param = choix.ilsr_pairwise(
                len(ordered) + 1,
                pairs,
                alpha=0.0001,
                max_iter=200)

        # TODO(sethtroisi): What should penalty be?
        hessian = choix.opt.PairwiseFcts(pairs, penalty=1).hessian(ilsr_param)
        std_err = np.sqrt(np.diagonal(np.linalg.inv(hessian)))

        # Elo conversion
        elo_mult = 400 / math.log(10)

        min_rating = min(ilsr_param)
        ratings = {}
        for model, param, err in zip(ordered, ilsr_param, std_err):
            ratings[model] = (elo_mult * (param - min_rating), elo_mult * err)

        return ratings
