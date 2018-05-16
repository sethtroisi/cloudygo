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
import math
import os
import re
import time
import zlib
from collections import Counter, defaultdict

import choix
import numpy as np

from . import sgf_utils


class CloudyGo:
    SALT_MULT = 10 ** 6
    GAME_TIME_MOD = 10 ** 8 # Most of a year in seconds
    GAME_POD_MULT = 10 ** 8
    GAME_BUCKET_MULT = 10 ** 3

    # set by __init__ but treated as constant
    INSTANCE_PATH = None
    DATA_DIR = None

    # set to 'minigo-pub' or similiar to serve debug games from the cloud.
    DEBUG_GAME_CLOUD_BUCKET = 'minigo-pub'

    DEFAULT_BUCKET = 'v7-19x19'

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


    def all_games(self, bucket, model, debug = False):
        game_type = 'full' if debug else 'clean'
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
        assert re.match('^[a-zA-Z_]*$', table), table
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
        game, name = filename.split('-', 1)
        pod = name.split('-')[-1][:-4]

        game_num = int(game) % CloudyGo.GAME_TIME_MOD

        pod_num = int(pod, 36)
        assert pod_num < CloudyGo.GAME_POD_MULT
        game_num = game_num * CloudyGo.GAME_POD_MULT + pod_num

        bucket_num, rem = divmod(bucket_salt, CloudyGo.SALT_MULT)
        assert bucket_num < CloudyGo.GAME_BUCKET_MULT, bucket_salt
        assert rem == 0, bucket_salt
        game_num = game_num * CloudyGo.GAME_BUCKET_MULT + bucket_num

        return game_num


    @staticmethod
    def get_eval_parts(filename):
        # 1519793529-000366-immune-elk-vs-000333-hot-pika-0.sgf
        assert filename.endswith('.sgf'), filename
        parts = re.split(r'[._-]+', filename)
        nums = [int(part) for part in parts if part.isnumeric()]
        assert len(nums) == 4, '{} => {}'.format(filename, parts)
        return nums


    @staticmethod
    def get_eval_num(filename):
        parts = CloudyGo.get_eval_parts(filename)
        sep = 1000
        return parts[0] * sep ** 3 + \
               parts[1] * sep ** 2 + \
               parts[2] * sep ** 1 + \
               parts[3]

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
        game_stats = []
        for filename in filenames:
            game_num = CloudyGo.get_game_num(bucket_salt, filename)
            game = self.query_db(
                'SELECT * FROM games WHERE game_num = ?',
                (game_num,))
            if len(game) == 0:
                continue
            games.append(game[0])

            # NOTE: if table is altered * may return unexpected order
            stats = self.query_db(
                'SELECT * FROM game_stats WHERE game_num = ?',
                (game_num,))
            if len(stats) == 0:
                game_stats.append(None)
            else:
                game_stats.append(stats[0])

        return games, game_stats


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

        public_path = os.path.join(model, view_type, filename)
        file_path = os.path.join(self.sgf_path(bucket), public_path)

        if view_type == 'eval':
            file_path = os.path.join(
                self.data_path(bucket), view_type, filename)

        data = ''
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                data = f.read()
        else:
            if 'full' in view_type and CloudyGo.DEBUG_GAME_CLOUD_BUCKET:
                data = self.__get_gs_game(bucket, model, filename, view_type)
                if data:
                    return data, view_type

            # Full games get deleted after X time, fallback to clean
            if 'full' in view_type:
                new_type = view_type.replace('full', 'clean')
                return self.get_game_data(bucket, model, filename, new_type)

        return data, view_type


    def get_existing_games(self, table, model_id):
        query = 'select game_num from {} where model_id = ?'.format(
            re.escape(table))
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


    def get_favorite_openings(self, model_id, num_stats_games):
        favorite_openings = self.query_db(
            'SELECT SUBSTR(early_moves_canonical,'
            '              0, instr(early_moves_canonical, ";")),'
            '       count(*)'
            'FROM game_stats WHERE model_id = ? '
            'GROUP BY 1 ORDER BY 2 DESC LIMIT 16',
            (model_id,))

        return [(move, round(100 * count / num_stats_games))
            for move, count in favorite_openings
                if 100 * count >= num_stats_games]


    #### PAGES ####


    def update_models(self, bucket, partial=False):
        model_filenames = glob.glob(
            os.path.join(self.model_path(bucket), '*.meta'))

        existing = self.get_models(bucket)

        model_inserts = []
        model_stat_inserts = []
        for model_filename in sorted(model_filenames):
            raw_name = os.path.basename(model_filename).replace('.meta', '')

            model_num, model_name = raw_name.split('-', 1)
            model_id = CloudyGo.bucket_salt(bucket) + int(model_num) # unique_id

            last_updated = int(time.time())
            creation = int(os.path.getmtime(model_filename))
            training_time_m = 120

            num_games = self.query_db(
                'SELECT count(*) from games WHERE model_id = ?',
                (model_id,))
            num_games = num_games[0][0]

            num_stats_games = self.query_db(
                'SELECT count(*) from game_stats WHERE model_id = ?',
                (model_id,))
            num_stats_games = num_stats_games[0][0]

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

            if partial and any(test[0] == model_id for test in existing):
                continue

            currently_processed = self.query_db(
                'SELECT max(stats_games) FROM model_stats WHERE model_id = ?',
                (model_id,))

            currently_processed = currently_processed[0][0] or 0
            if num_stats_games == 0 or num_stats_games == currently_processed:
                continue

            opening_name = str(model_id) + '-favorite-openings.png'
            opening_file = os.path.join(
                self.INSTANCE_PATH, 'openings', opening_name)
            opening_sgf  = sgf_utils.board_png(
                CloudyGo.bucket_to_board_size(bucket),
                '', #setup
                self.get_favorite_openings(model_id, num_stats_games),
                opening_file,
                force_refresh=True)

            games = self.query_db(
                'SELECT * from game_stats WHERE model_id = ?',
                (model_id,))

            for perspective in ['all', 'black', 'white']:
                is_all = perspective == 'all'
                is_black = perspective == 'black'

                # ASSUMPTION: every game has a result
                wins = games
                if not is_all:
                    wins = [game for game in games if game[2] == is_black]

                wins_by_resign = len([1 for game in wins if '+R' in game[3]])
                sum_wins_result = sum(float(game[3][2:]) for game in wins
                    if '+R' not in game[3])

                resign_rates = Counter(game[14] for game in wins)
                resign_rates.pop(-1, None) # remove -1 if it's present
                if len(resign_rates) != 1:
                    if perspective == 'all':
                        print('{} has multiple Resign rates: {}'.format(
                            raw_name, resign_rates))
                    holdouts = []
                    bad_resigns = -1
                else:
                    resign_rate = max(resign_rates.keys())

                    holdouts = [game for game in wins
                                    if abs(game[14]) > abs(resign_rate)]
                    holdout_resigns = [game for game in holdouts if '+R' in game[3]]
                    assert len(holdout_resigns) == 0, holdout_resigns

                    bad_resigns = 0
                    for game in holdouts:
                        black_won = game[2]

                        black_would_resign = game[15] < resign_rate
                        white_would_resign = game[16] < resign_rate

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

                    sum(game[5] for game in wins), # num_moves
                    sum(game[8] + game[9] for game in wins), # both sides visits
                    sum(game[10] + game[11] for game in wins), # both sides early visits
                    sum(game[12] + game[13] for game in wins), # both sides unluckiness
                    opening_sgf if is_all else '', # favorite_openings
                ))

        db = self.db()

        cur = db.executemany(
            'DELETE FROM models WHERE model_id = ?',
            [(model[0],) for model in model_inserts])
        removed = cur.rowcount

        cur = db.executemany(
            'DELETE FROM model_stats WHERE model_id = ?',
            [(model[0],) for model in model_stat_inserts])
        removed_stats = cur.rowcount

        self.insert_rows_db('models', model_inserts)
        self.insert_rows_db('model_stats', model_stat_inserts)

        db.commit()

        print('updated:  {} existing, {}|{} removed, {}|{} inserts'.format(
            len(existing),
            removed, removed_stats,
            len(model_inserts), len(model_stat_inserts)))
        return len(model_inserts) + len(model_stat_inserts)


    @staticmethod
    def process_game(data):
        game_path, stats, game_num, filename, model_id = data
        if stats:
            result = sgf_utils.parse_game(game_path)
            if not result: return None
            record = (game_num, model_id,) + result
        else:
            result = sgf_utils.parse_game_simple(game_path)
            if not result: return None
            record = (game_num, filename, model_id,) + result

        return record


    def update_games(self, bucket, stats, max_inserts, min_model):
        # This is REALLY SLOW because it's potentially >1M items
        # loop by model to avoid huge globs and commits
        skipped = 0
        updates = 0
        results = []

        bucket_salt = CloudyGo.bucket_salt(bucket)
        table = 'game_stats' if stats else 'games'

        for model in self.get_models(bucket):
            if min_model > model[4]:
                skipped += 1
                continue

            # Check if directories mtime is recent, if not skip
            base = os.path.join(self.sgf_path(bucket), model[2])
            times = [0]
            for test_d in [os.path.join(base, d) for d in ('clean', 'full')]:
                if os.path.exists(test_d):
                    times.append(os.path.getmtime(test_d))
            if min_model == 0 and max(times) < model[5] - 86400:
                continue

            model_id = model[0]
            existing = self.get_existing_games(table, model_id)

            games_added = set()
            games_to_process = []
            for game_path in self.all_games(bucket, model[2], debug=stats):
                filename = os.path.basename(game_path)
                game_num = CloudyGo.get_game_num(bucket_salt, filename)

                if game_num in existing: continue

                assert game_num not in games_added, (game_num, game_path)
                games_added.add(game_num)

                games_to_process.append(
                    (game_path, stats, game_num, filename, model_id))

                if len(games_to_process) >= max_inserts:
                    break

            if self.pool:
                new_games = self.pool.map(CloudyGo.process_game, games_to_process)
            else:
                new_games = map(CloudyGo.process_game, games_to_process)

            new_games = list(filter(None.__ne__, new_games))
            total_games = len(existing) + len(new_games)

            self.insert_rows_db(table, new_games)

            update_model_query = \
                'UPDATE models SET {} = ? WHERE model_id = ?'.format(
                        re.escape('num_stats_games' if stats else 'num_games'))
            self.db().execute(update_model_query, (total_games, model_id,))

            self.db().commit()

            result = '{}: {} existing, {} inserts'.format(
                model[2], len(existing), len(new_games))
            if len(new_games):
                print (result)

            updates += len(new_games)
            results.append(result)

        skipped_text = 'skipped {} models'.format(skipped)
        results.append(skipped_text)
        return updates, '<br>'.join(reversed(results))


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
    def eval_record(data):
        eval_path, filename, eval_num, model_id_1, model_id_2 = data
        result = sgf_utils.parse_game_simple(eval_path)
        if not result: return None
        return (eval_num, filename, model_id_1, model_id_2) + result


    def update_eval_games(self, bucket):
        if not os.path.exists(self.eval_path(bucket)):
            return 0

        bucket_salt = CloudyGo.bucket_salt(bucket)

        existing = self.get_existing_eval_games(bucket)
        evals_to_process = []

        eval_games = glob.glob(os.path.join(self.eval_path(bucket), '*.sgf'))
        # sort by newest first
        eval_games = sorted(eval_games, reverse=True)
        for eval_path in eval_games:
            filename = os.path.basename(eval_path)
            eval_num = CloudyGo.get_eval_num(filename)

            if eval_num in existing: continue

            eval_parts = CloudyGo.get_eval_parts(filename)

            evals_to_process.append(
                (eval_path,
                 filename,
                 eval_num,
                 bucket_salt + eval_parts[1],
                 bucket_salt + eval_parts[2]))

        new_evals = []
        if len(evals_to_process) > 0:
            if self.pool:
                new_evals = self.pool.map(CloudyGo.eval_record, evals_to_process)
            else:
                new_evals = map(CloudyGo.eval_record, evals_to_process)

            broken = new_evals.count(None)
            new_evals = list(filter(None.__ne__, new_evals))

            if broken > 10:
                print ("{} Broken games".format(broken))

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
            # TODO(sethtroisi): remove assumption white was first player
            white, black, black_won = d

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
        min_id = min(model_nums)
        max_id = max(model_nums)

        def ilsr_data(eval_game):
            # TODO actually parse PW and PB
            p1, p2, black_won = eval_game
            p1 -= min_id
            p2 -= min_id
            assert 0 <= p1 <= 1000
            assert 0 <= p2 <= 1000

            return (p2, p1) if black_won else (p1, p2)

        pairs = list(map(ilsr_data, eval_games))
        ilsr_param = choix.ilsr_pairwise(
                max_id - min_id + 1,
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
        for num in model_nums:
            index = num - min_id
            ratings[num] = (elo_mult * (ilsr_param[index] - min_rating), elo_mult * std_err[index])

        return ratings
