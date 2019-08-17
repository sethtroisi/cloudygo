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

import itertools
import json
import heapq
import operator
import os
import numpy as np
import random
import re
import pickle
import stat
import sqlite3
import time

from collections import defaultdict
from datetime import datetime

from flask import Flask, g
from flask import render_template
from flask import send_from_directory, url_for, jsonify
from flask import request, Response
from werkzeug.contrib.cache import SimpleCache

from . import sgf_utils
from .cloudygo import CloudyGo


app = Flask(__name__)
app.jinja_env.trim_blocks = True
app.jinja_env.lstrip_blocks = True

# Requires Apache support see:
# https://stackoverflow.com/a/27303164/459714
app.use_x_sendfile = True and not app.debug

cache = SimpleCache()

LOCAL_DATA_DIR = os.path.join(app.instance_path, 'data')
LOCAL_EVAL_DIR = os.path.join(app.instance_path, 'eval')
DATABASE_PATH = os.path.join(app.instance_path, 'clouds.db')

RANDOMIZE_GAMES = True
MAX_GAMES_ON_PAGE = 100

CONVERTED_SUFFIX = "_converted.txt.gz"

#### DB STUFF ####


def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE_PATH)
        db.row_factory = sqlite3.Row
    return db


@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()


print('Setting up Cloudy')
cloudy = CloudyGo(
    app.instance_path,
    LOCAL_DATA_DIR,
    get_db,
    cache,
    None,  # multiprocessing pool
)

#### UTILS ####

def is_naughty(filepath, basepath, suffix):
    base_dir_abs = os.path.abspath(basepath)
    file_path_abs = os.path.abspath(filepath)
    return not (file_path_abs.startswith(base_dir_abs) and
                file_path_abs.endswith(suffix) and
                os.path.exists(file_path_abs))


def get_bool_arg(name, args):
    value = args.get(name, 'false').lower()
    return value not in ('f', 'false')


@app.template_filter('strftime')
def _jinja2_filter_strftime(time, fmt=None):
    date = datetime.fromtimestamp(int(time))
    tformat = '%Y-%m-%d %H:%M'
    return date.strftime(tformat)

# TODO(seth): move file servers here

#### PAGES ####

@app.route('/results')
@app.route('/RESULTS')
def results():
    return render_template('README.html')


@app.route('/sprt')
@app.route('/calc')
def SPRT():
    # Consider adding params for a,b, wins, losses, ...
    return render_template('sprt.html')


@app.route('/213-secret-site-nav')
@app.route('/<bucket>/213-secret-site-nav')
def debug(bucket=CloudyGo.DEFAULT_BUCKET):
    db_mtime = os.path.getmtime(DATABASE_PATH)
    secret_vars = [
        'app.instance_path: ' + app.instance_path,
        'DEFAULT_BUCKET: ' + cloudy.DEFAULT_BUCKET,
        '[UNUSED] app.root_path: ' + app.root_path,
        'DATABASE_FILE: {}'.format(os.path.exists(DATABASE_PATH)),
        'DATABASE_RAW: {}'.format(os.stat(DATABASE_PATH).st_size),
        'DATABASE M_TIME: {}'.format(datetime.fromtimestamp(db_mtime)),
    ]

    # try to filter some of the rsync lines
    patterns = list(map(re.compile, [
        # >1k, > 100, or 0 file all get filtered
        r'.*\[[0-9.k]* files\]',  # Not started just counting how many
        r'.*\[[0-9.k]*/[0-9.]*k files\]\[.*Done',
        r'.*\[[1-9][0-9]{2}/.* files\]\[.*Done',
        r'Copying gs://.*/sgf/.*sgf',
        r'[0-9]{3,}it ',
        r'^.{0,3}$',
        r'^idx \d* already processed',
        r'.*[0-9]{2}it/s',  # tqdm output
        r'.*xfr#[0-9]+,',    # rsync --info=progress2
    ]))

    def not_boring_line(line):
        if len(line) > 500:
            return False
        return random.randrange(150) == 0 or \
            all(not pattern.match(line) for pattern in patterns)

    log_files = ['cloudy-rsync-cron.log', 'cloudy-eval.log',
                 'cloudy-sync-all.log', 'cloudy-heatmap.log',
    ]
    log_datas = []
    for log in log_files:
        log_filename = os.path.join(app.instance_path, 'debug', log)
        if not os.path.exists(log_filename):
            print("log: {} does not exist".format(log_filename))
            continue

        log_lines = []
        full_count = 0
        with open(log_filename, 'r', encoding='utf-8') as log_file:
            log_data = set()
            for line in log_file.readlines():
                full_count += 1
                if line not in log_data and not_boring_line(line):
                    log_lines.append(line)
                    log_data.add(line)

        log_datas.append((log_filename, full_count, log_lines))

    return render_template(
        'secret-site-nav.html',
        bucket=bucket,
        logs=log_datas,
        full_count=full_count,
        secret_vars=secret_vars,
    )


@app.route('/openings/<filename>')
def opening_image(filename):
    folder = os.path.join(app.instance_path, 'openings')
    path = os.path.join(folder, filename)
    if is_naughty(path, app.instance_path, '.png'):
        return ''

    return send_from_directory(
        folder,
        filename,
        cache_timeout=60*60)


@app.route('/photos/thumbs/<name>')
def model_thumb(name):
    folder = os.path.join(app.instance_path, 'photos', 'thumbs')
    path = os.path.join(folder, name)
    if is_naughty(path, app.instance_path, '.jpg'):
        return ''

    return send_from_directory(
        folder,
        name,
        cache_timeout=60*60)


def _fstat_dir(directory, top_dir):
    if not os.path.isdir(directory):
        return []

    files = os.listdir(directory)
    f_stats = []
    for f in files:
        f_stats.append([
            os.path.join(top_dir, f),
            os.stat(os.path.join(directory, f)),
        ])
    f_stats = sorted(f_stats, key=lambda f: f[1].st_mtime, reverse=True)
    return f_stats

@app.route('/converted_model/')
@app.route('/converted_model/<path:filename>/')
def converted_model(filename=""):
    if filename == "":
        filename = os.path.join(CloudyGo.DEFAULT_BUCKET, "models")
    filepath = os.path.join(LOCAL_DATA_DIR, filename)

    if os.path.isfile(filepath):
        if is_naughty(filepath, LOCAL_DATA_DIR, ".txt.gz"):
            return 'Not Found'

        return send_from_directory(
            LOCAL_DATA_DIR,
            filename,
            as_attachment=True)

    if is_naughty(filepath, LOCAL_DATA_DIR, "models"):
        return 'must end in models'
    if not os.path.isdir(filepath):
        return ''

    f_stats = _fstat_dir(filepath, filename)
    f_stats = [(f,stats) for f,stats in f_stats if
        f.endswith(CONVERTED_SUFFIX)]

    return render_template(
        'fileslist.html',
        navbar_title='Minigo Models Converted to Leela-Zero weights',
        header='{} Found'.format(len(f_stats)),
        serve_func='converted_model',
        files=f_stats)


@app.route('/ringmaster/')
@app.route('/ringmaster/<path:filename>/')
def ctl_file(filename=""):
    folder = os.path.join(app.instance_path, 'ringmaster')
    filepath = os.path.join(folder, filename)
    if is_naughty(filepath, app.instance_path, ''):
        return ''

    if any(filename.endswith('.' + ext) for ext in
               ['ctl', 'report', 'hist', 'log']):
        return send_from_directory(
            folder,
            filename,
            mimetype='text/plain',
            cache_timeout=15*60)

    if filepath.endswith('.sgf') and os.path.isfile(filepath):
        with open(filepath) as f:
            data = f.read()
        return render_game(
            bucket="ringmaster",
            model_name="",
            data=data,
            filename="",
            force_full=True)

    if not (filename == "" or filename.endswith(".games")):
        return 'Restricted'

    if not os.path.isdir(filepath):
        return ''

    f_stats = _fstat_dir(filepath, filename)

    return render_template(
        'fileslist.html',
        navbar_title='Ringmaster CTL Files({})'.format(len(f_stats)),
        header='Various ringmaster files (updated sporadically).',
        serve_func='ctl_file',
        files=f_stats)


@app.route('/<bucket>/<model_name>/eval/<path:filename>')
def eval_view(bucket, model_name, filename):
    return game_view(
        bucket,
        model_name,
        filename,
    )

def render_game(bucket, model_name, data, filename="",
                force_full=False, render_sorry=False):
    is_raw = get_bool_arg('raw', request.args)
    if is_raw:
        if request.args.get('raw', '') == 'sgf':
            return Response(data, mimetype='application/x-go-sgf')
        return sgf_utils.pretty_print_sgf(data)

    # 3200 > 500 * 'B[aa];'
    player_evals = []
    if len(data) > 3200:
        try:
            # NOTE: evals are printed ~near~ the move they are for but plus or
            # minus one because of 2*m+1 below.
            _, comments = sgf_utils.raw_game_data(filename, data)
            evals = [comment[2][0] for comment in comments]
            for m, (b_eval, w_eval) in enumerate(zip(evals[::2], evals[1::2])):
                player_evals.append((2 * m + 1, b_eval, w_eval))
        except Exception as e:
            print("Failed to eval parse:", bucket, model_name)
            print(e)
            pass

    return render_template(
        'game.html',
        bucket=bucket,
        model=model_name,
        data=data,
        player_evals=player_evals,
        filename=filename,
        force_full=force_full or len(player_evals) > 0,
        render_sorry=False,
    )


@app.route('/<bucket>/<model_name>/game/<path:filename>')
@app.route('/<bucket>/<model_name>/clean/<path:filename>')
@app.route('/<bucket>/<model_name>/full/<path:filename>')
def game_view(bucket, model_name, filename):
    view_type = request.args.get('type')
    if not view_type:
        path = re.search(r'/(clean|full|eval)/', request.base_url)
        if path:
            view_type = path.group(1)
        else:
            view_type = 'clean'
    assert view_type in ('clean', 'eval', 'full'), view_type

    data, game_view = cloudy.get_game_data(
        bucket, model_name, filename, view_type)

    render_sorry = game_view != view_type

    # HACK: we'd like all eval games to be full in the future
    is_full_eval = 'cc-evaluator' in filename

    return render_game(bucket, model_name, data,
        filename=filename,
        force_full=is_full_eval,
    )


@app.route('/sgf/<path:filename>')
def send_game(filename):
    path = os.path.join(LOCAL_DATA_DIR, filename)
    if is_naughty(path, LOCAL_DATA_DIR, ''):
        return ''

    if not os.path.exists(path):
        return 'Not Found'


    mimetypes = {
        '.png': 'image/png',
        '.sgf': 'xapplication/x-go-sgf',
    }

    mimetype = mimetypes.get(path[-4:], None)

    if mimetype:
        return send_from_directory(
            LOCAL_DATA_DIR,
            filename,
            mimetype=mimetype,
            cache_timeout=30*60)
    return 'Not Found'

@app.route('/secret-pro-games/<path:filename>')
def pro_game_view(filename):
    base_dir = os.path.join(app.instance_path, 'pro')
    file_path = os.path.join(base_dir, filename)

    # make sure filename is in pro directory
    if is_naughty(file_path, base_dir, '.sgf'):
        return 'being naughty?'

    data = ''
    with open(file_path, 'r') as f:
        data = f.read()

    return render_game(
        bucket=CloudyGo.DEFAULT_BUCKET,  # Any value will do
        model_name='100',      # needs some value
        data=data,
        filename=filename,
    )


def parse_fig3_data(bucket):
    fig3_json = '{"acc":{}, "mse":{}, "num":{}}'

    json_file = os.path.join(LOCAL_EVAL_DIR, bucket, "fig3.json")
    if os.path.exists(json_file):
        with open(json_file, "r") as f:
            fig3_json = f.read()

    fig3_data = []
    try:
        fig3_json = json.loads(fig3_json)
        for k, v in fig3_json["num"].items():
            acc = fig3_json["acc"][k]
            mse = fig3_json["mse"][k]
            fig3_data.append((v, acc, mse))
    except Exception as e:
        print("Error parsing fig3 json:", e)

    return sorted(fig3_data)


@app.route('/<bucket>/figure-three')
def figure_three(bucket):
    figure_data = None
    figure_three_data = []

    # This list is a reasonable proxy for 'all' minigo runs.
    for other_bucket in CloudyGo.MINIGO_TS:
        key = '{}/fig3-json'.format(other_bucket)
        fig3_data = cache.get(key)
        if fig3_data is None:
            fig3_data = parse_fig3_data(other_bucket)
            cache.set(key, fig3_data, timeout=5 * 60)

        if other_bucket == bucket:
            figure_data = fig3_data

        for values in fig3_data:
            if values[0] % 2 == 0:
                figure_three_data.append((other_bucket,) + values)

    return render_template('figure-three.html',
                           bucket=bucket,
                           fig3_data=figure_data,
                           figure_three_data=figure_three_data,
                           )

@app.route('/<bucket>/openings.html')
def joseki_openings(bucket):
    directory = os.path.join(app.instance_path, 'joseki')
    f = bucket[:3] + "openings.html"
    if not os.path.exists(os.path.join(directory, f)):
        return 'Josekis not calculated for this run'

    # For details on how these were generated see
    # https://github.com/tensorflow/minigo/pull/860

    return send_from_directory(directory, f, cache_timeout=10*60)


@app.route('/site-nav')
@app.route('/<bucket>/site-nav')
def site_nav(bucket=CloudyGo.DEFAULT_BUCKET):
    model_name = cloudy.get_newest_model_num(bucket)

    return render_template(
        'site-nav.html',
        bucket=bucket,
        model=model_name)


@app.route('/')
@app.route('/<bucket>/')
@app.route('/<bucket>/models/')
def models_details(bucket=CloudyGo.DEFAULT_BUCKET):
    models = sorted(cloudy.get_models(bucket))[::-1]
    run_data = cloudy.get_run_data(bucket)
    total_games = sum((m[9] for m in models))

    while len(models):
        # If model has games or isn't recent
        if models[0][9] > 0 or (time.time() - models[0][7]) > 3600:
            break
        models.pop(0)

    # Limit to recent and mod ten models
    if len(models) > 120:
        trim_count = len(models) - 120
        skippable = sorted(m[0] for m in models if m[0] % 10 != 0)
        to_skip = set(skippable[:trim_count])
        models = [m for m in models if m[0] not in to_skip]

    # Convert tuples to mutable lists (so that timestamps can be datified).
    models = [list(m) for m in models]

    last_update = max((m[6] for m in models), default=0)
    for m in models:
        # creation timestamp
        m[7] = CloudyGo.time_stamp_age(m[7])

    return render_template(
        'models.html',
        bucket=bucket,
        run_data=run_data,
        models=models,
        last_update=last_update,
        total_games=total_games,
    )


@app.route('/<bucket>/graphs')
def models_graphs(bucket):
    model_limit = int(request.args.get('last_n', 300))
    model_range = CloudyGo.bucket_model_range(bucket)

    key = '{}/graphs/{}'.format(bucket, model_limit)
    graphs = cache.get(key)
    if graphs is None:
        win_rate = cloudy.bucket_query_db(
            bucket,
            'SELECT model_id % 1000000, round(1.0*wins/num_games,3)',
            'model_stats', 'WHERE perspective = "black"', 1, model_limit)

        bad_resign_rate = cloudy.bucket_query_db(
            bucket,
            'SELECT model_id % 1000000, round(1.0*bad_resigns/hold_out_resigns,3)',
            'model_stats',
            'WHERE hold_out_resigns > 0 and perspective = "all" ',
            1,
            model_limit)

        newest_model = cloudy.get_newest_model_num(bucket) + model_range[0]
        alternative_resign_rate = cloudy.query_db(
            'SELECT model_id % 1000000, '
            '   black_won * -bleakest_eval_black + '
            '   (1-black_won) * bleakest_eval_white '
            'FROM games '
            'WHERE (model_id BETWEEN ? AND ?)',
            (max(model_range[0], newest_model - 30), newest_model))

        # model, rate below which would fail
        bad_resign_thresh = defaultdict(list)
        for model, bleakest in alternative_resign_rate:
            if bleakest == None:
                # TODO: https://github.com/tensorflow/minigo/issues/666
                continue
            bad_resign_thresh[model].append(bleakest)

        # calculate some percentages
        for m in bad_resign_thresh:
            bleak = sorted(bad_resign_thresh[m])
            percents = [(p, round(np.percentile(bleak, 100 - p), 3))
                        for p in range(10)]
            bad_resign_thresh[m] = percents
        bad_resign_thresh = sorted(bad_resign_thresh.items())

        game_length_simple = cloudy.bucket_query_db(
            bucket,
            'SELECT model_id % 1000000, round(1.0*num_moves/num_games,3)',
            'model_stats', 'WHERE perspective = "all"', 1, model_limit)

        num_games = cloudy.bucket_query_db(
            bucket,
            'SELECT model_id % 1000000, num_games, stats_games',
            'model_stats', 'WHERE perspective = "all"', 1, model_limit)

        games_per_day = cloudy.bucket_query_db(
            bucket,
            'SELECT date(creation, "unixepoch"), sum(num_games)',
            'models', '', 1)

        num_visits = cloudy.bucket_query_db(
            bucket,
            'SELECT model_id % 1000000, number_of_visits/stats_games',
            'model_stats', 'WHERE perspective = "all"', 1, model_limit)

        sum_unluck = cloudy.bucket_query_db(
            bucket,
            'SELECT model_id % 1000000, round(sum_unluckiness/stats_games,2)',
            'model_stats', 'WHERE perspective = "all"', 1, model_limit)

        ratings_range = (model_range[0],
                         model_range[0] + CloudyGo.CROSS_EVAL_START - 10)
        half_curve_rating_delta = cloudy.query_db(
            'SELECT '
            '    model_id_1 - model_id_2, '
            '    100 * avg((m1_black_wins + m1_white_wins) / ('
            '           (m1_black_games + m1_white_games + 0.001))) '
            'FROM eval_models m '
            'WHERE (model_id_1 BETWEEN ? AND ?) AND '
            '      (model_id_2 BETWEEN ? AND ?) AND '
            '       model_id_2 != 0 '
            'GROUP BY 1 ORDER BY 1 asc',
            ratings_range + ratings_range)

        rating_delta = cloudy.query_db(
            'SELECT m.model_id_1 % 1000000, m.rankings - m2.rankings '
            'FROM eval_models m INNER JOIN eval_models m2 '
            'WHERE m.model_id_2 = 0 AND m2.model_id_2 = 0 '
            '   AND m.model_id_1 - 1 = m2.model_id_1 '
            '   AND (m.model_id_1 BETWEEN ? AND ?) '
            'ORDER BY m.model_id_1 desc LIMIT ?',
            ratings_range + (model_limit,))
        rating_delta = list(reversed(rating_delta))

        graphs = (win_rate,
                  bad_resign_rate, bad_resign_thresh,
                  game_length_simple,
                  num_games, games_per_day,
                  num_visits,
                  rating_delta,
                  sum_unluck,
                  half_curve_rating_delta)
        cache.set(key, graphs, timeout=10 * 60)
    else:
        win_rate, \
            bad_resign_rate, bad_resign_thresh, \
            game_length_simple, \
            num_games, games_per_day, \
            num_visits, \
            rating_delta, \
            sum_unluck, \
            half_curve_rating_delta = graphs

    return render_template('models-graphs.html',
                           bucket=bucket,
                           win_rate=win_rate,
                           bad_resign_rate=bad_resign_rate,
                           bad_resign_thresh=bad_resign_thresh,
                           game_len_simple=game_length_simple,
                           num_games=num_games,
                           games_per_day=games_per_day,
                           num_visits=num_visits,
                           rating_delta=rating_delta,
                           sum_unluck=sum_unluck,
                           win_rate_curve_delta=half_curve_rating_delta,
                           )


@app.route('/<bucket>/graphs-sliders')
def models_graphs_sliders(bucket):
    key = '{}/graphs-sliders'.format(bucket)
    graphs = cache.get(key)
    if graphs is None:
        # Divide by four to help avoid the 'only black can win on even moves'
        game_length = cloudy.bucket_query_db(
            bucket,
            'SELECT model_id % 1000000, black_won, 4*(num_moves/4), count(*)',
            'games', 'WHERE model_id % 1000000 >= 50 ', 3, limit=20000)

        sum_unluck_per = cloudy.bucket_query_db(
            bucket,
            'SELECT '
            '   model_id % 1000000, black_won, '
            '   round(100 * (unluckiness_black - unluckiness_white) / '
            '       (unluckiness_black + unluckiness_white), 0), '
            '   count(*) ',
            'games', '', 3, limit=20000)

        picture_sliders = []

        # TODO replace with list of SGFs
        models = sorted(cloudy.get_models(bucket))
        for model in models:
            model_id = str(model[0])
            opening = model_id + '-favorite-openings.png'
            policy = model_id + '-policy-empty.png'

            picture_sliders.append((
                model[0] % CloudyGo.SALT_MULT,
                url_for('.opening_image', filename=opening),
                url_for('.opening_image', filename=policy)
            ))

        graphs = (game_length, sum_unluck_per, picture_sliders)
        cache.set(key, graphs, timeout=10 * 60)
    else:
        game_length, sum_unluck_per, picture_sliders = graphs

    return render_template('models-graphs-sliders.html',
                           bucket=bucket,
                           game_length=game_length,
                           sum_unluck_per=sum_unluck_per,
                           picture_sliders=picture_sliders,
                           )


@app.route('/<bucket>/model_comparison/policy/<model_name_a>/<model_name_b>')
@app.route('/<bucket>/model_comparison/pv/<model_name_a>/<model_name_b>')
def position_comparison(bucket, model_name_a, model_name_b):
    model_a, _ = cloudy.load_model(bucket, model_name_a)
    model_b, _ = cloudy.load_model(bucket, model_name_b)
    if model_a is None or model_b is None:
        return 'Model {} or {} not found'.format(model_name_a, model_name_b)

    rule_group = 'policy' if '/policy/' in request.url_rule.rule else 'pv'
    arg_group = request.args.get('group', None)
    group = arg_group or rule_group

    count, data = cloudy.get_position_sgfs(bucket, [model_a[0], model_b[0]])

    return render_template('position-comparison.html',
                           bucket=bucket,
                           model_a=model_a,
                           model_b=model_b,
                           group=group,
                           sgfs=data,
                           )


@app.route('/<bucket>/models_evolution/')
def models_evolution(bucket):
    count, sgfs = cloudy.get_position_sgfs(bucket)
    return render_template('position-evolution.html',
                           bucket=bucket,
                           sgfs=sgfs,
                           count=count,
                           )


def get_eval_data(bucket):
    model_range = CloudyGo.bucket_model_range(bucket)
    bucket_salt = CloudyGo.bucket_salt(bucket)

    eval_models = cloudy.query_db(
        'SELECT * FROM eval_models '
        'WHERE (model_id_1 BETWEEN ? AND ?) AND model_id_2 = 0 '
        '      AND games >= 4 '
        'ORDER BY model_id_1 desc',
        model_range)

    # Only count black games so not to double count.
    total_games = sum(m[5] for m in eval_models)
    num_to_name = cloudy.get_model_names(model_range)
    return model_range, bucket_salt, eval_models, total_games, num_to_name


@app.route('/<bucket>/eval-graphs')
def eval_graphs(bucket):
    model_range, bucket_salt, eval_models, total_games, num_to_name = \
        get_eval_data(bucket)

    if len(eval_models) < 2:
        return render_template('models-eval-empty.html',
                               bucket=bucket, total_games=total_games)

    # Replace model_id_2 with name
    def eval_model_transform(m):
        model_id = m[0]
        num = model_id - bucket_salt
        name = num_to_name.get(model_id, str(num))
        return (bucket, num, name) + m[2:]

    eval_models = list(map(eval_model_transform, eval_models))

    # If directory has auto-names then it's a dir_eval and not a run_eval
    max_model_id = max((m[1] for m in eval_models), default=0)
    is_sorted = get_bool_arg('sorted', request.args)
    is_sorted = is_sorted or max_model_id >= CloudyGo.CROSS_EVAL_START

    sort_by_rank = operator.itemgetter(3)
    eval_models_by_rank = sorted(eval_models, key=sort_by_rank, reverse=True)

    top_ten_threshold = eval_models_by_rank[10][3]

    older_newer_winrates = cloudy.query_db(
        'SELECT model_id_1 % 1000000, '
        '       sum((model_id_1 > model_id_2) * (m1_black_wins+m1_white_wins)), '
        '       sum((model_id_1 > model_id_2) * games), '
        '       sum((model_id_1 < model_id_2) * (m1_black_wins+m1_white_wins)), '
        '       sum((model_id_1 < model_id_2) * games) '
        'FROM eval_models '
        'WHERE (model_id_1 BETWEEN ? AND ?) AND model_id_2 != 0 '
        'GROUP BY 1 ORDER BY 1 asc',
        model_range)

    return render_template('models-eval.html',
                           bucket=bucket,
                           is_sorted=is_sorted,
                           total_games=total_games,

                           models=eval_models,
                           sorted_models=eval_models_by_rank,
                           great_threshold=top_ten_threshold,

                           older_newer_winrates=older_newer_winrates,
                           )

@app.route('/all-eval-graphs')
def all_eval_graphs():
    bucket = request.args.get('bucket', 'cross-run-eval')
    bucket = bucket if bucket in CloudyGo.ALL_EVAL_BUCKETS else 'cross-run-eval'

    model_range, bucket_salt, eval_models, total_games, num_to_name = \
        get_eval_data(bucket)

    cross_run_models = cloudy.query_db(
        'SELECT * FROM eval_models '
        'WHERE (model_id_1 BETWEEN ? AND ?) AND model_id_2 != 0 '
        '      AND games >= 4 ',
        model_range)
    cross_run_games = sum(m[5] for m in cross_run_models
        if num_to_name[m[0]].split('/')[0] !=
           num_to_name[m[1]].split('/')[0])

    def eval_model_transform(m):
        model_id = m[0]
        name = num_to_name.get(model_id, str(model_id))

        test = re.split(r'[/-]', name)
        assert len(test) >= 4, (name, test)
        e_bucket = '-'.join(test[0:2])
        num = int(test[2])
        name = '-'.join(test[2:])

        e_bucket = e_bucket.replace('v9', 'v09')
        model_id -= bucket_salt

        metadata = (e_bucket, num, model_id, name)
        return metadata + m[2:]

    eval_models = list(map(eval_model_transform, eval_models))
    eval_models = sorted(eval_models, reverse=True)

    def sorted_by(column):
        return sorted(
            eval_models,
            key=operator.itemgetter(column),
            reverse=True)

    eval_models_by_rank = sorted_by(4)

    # Make sure each bucket has at least a couple
    eval_models_by_games = sorted_by(6)
    other_buckets = sorted(set(m[0] for m in eval_models_by_games))
    top_by_bucket = [[m for m in eval_models_by_games if m[0] == b][:3][::-1]
                        for b in other_buckets]
    eval_models_by_games = sum(top_by_bucket, [])

    return render_template('models-eval-cross.html',
                           bucket=bucket,
                           total_games=total_games,
                           cross_run_games=cross_run_games,
                           models=eval_models,
                           sorted_models=eval_models_by_rank,
                           well_played_models=eval_models_by_games[::-1],
                           )

@app.route('/<bucket>/eval-model/<model_name>')
def model_eval(bucket, model_name):
    bucket_salt = CloudyGo.bucket_salt(bucket)
    # TODO try and use model_name + name_to_model_id instead of model_id
    model, model_stats = cloudy.load_model(bucket, model_name)
    if model == None:
        try:
            model_id = bucket_salt + int(model_name)
            model = [model_id,
                     model_name, model_name, model_name,
                     bucket,
                     model_id, # num
                     0, 0, 0,  # last updated, creation, training_time
                     0, 0, 0]  # games, stat_games, eval_games
        except:
            return "Unsure of model id for \"{}\"".format(model_name)

    # Nicely format creation timestamp.
    model = list(model)
    model[7] = CloudyGo.time_stamp_age(model[7])[0] if model[7] else '???'

    eval_models = cloudy.query_db(
        'SELECT * FROM eval_models WHERE model_id_1 = ?',
        (model[0],))
    total_games = sum(e_m[2] for e_m in eval_models)

    if total_games == 0:
        return 'No games for ' + model_name

    model_range = CloudyGo.bucket_model_range(bucket)
    num_to_name = cloudy.get_model_names(model_range)

    overall = [e_m for e_m in eval_models if e_m[1] == 0][0]
    eval_models.remove(overall)
    overall = list(overall)
    overall[0] %= CloudyGo.SALT_MULT
    overall[1] = num_to_name.get(model[0], overall[0])
    rating = overall[2]

    # TODO: Probably should be something better
    rank = cloudy.query_db(
        'SELECT 1 + count(*) '
        'FROM eval_models '
        'WHERE rankings > ? AND '
        '      model_id_1 / 1000000 = ? AND '
        '      model_id_2 = 0',
        (rating, bucket_salt / CloudyGo.SALT_MULT))[0][0]

    updated = []
    played_better = 0
    later_models = [0, 0]
    earlier_models = [0, 0]
    for e_m in eval_models:
        # Make models more familiar
        cur_id = e_m[0] % CloudyGo.SALT_MULT
        other_id = e_m[1] % CloudyGo.SALT_MULT
        other_name = num_to_name.get(e_m[1], other_id)
        rating_diff = 2 * (e_m[2] - rating)

        updated.append((cur_id, other_id, other_name, rating_diff) + e_m[3:])

        # e_m[2] is average rating (of ours + theirs)
        if e_m[2] > rating:
            played_better += 1

        if e_m[1] < e_m[0]:
            earlier_models[0] += e_m[4]
            earlier_models[1] += e_m[6] + e_m[8]
        else:
            later_models[0] += e_m[4]
            later_models[1] += e_m[6] + e_m[8]

    eval_games = cloudy.query_db(
        'SELECT '
        '    model_id_1 % 1000000, '
        '    model_id_2 % 1000000, '
        '    filename, '
        '    black_won = (model_id_1 = ?) '
        'FROM eval_games '
        'WHERE model_id_1 = ? or model_id_2 = ? '
        'ORDER by filename',
        (model[0], model[0], model[0]))

    model[0] %= CloudyGo.SALT_MULT

    is_sorted = get_bool_arg('sorted', request.args)
    sort_by = operator.itemgetter(3 if is_sorted else 1)
    eval_models = sorted(updated, key=sort_by)

    return render_template('model-eval.html',
                           bucket=bucket,
                           is_sorted=is_sorted,
                           total_games=total_games,
                           model=model,
                           overall=overall,
                           rank=rank,
                           played_better=played_better,
                           later_models=later_models,
                           earlier_models=earlier_models,
                           eval_models=eval_models,
                           eval_games=eval_games,
    )


# Supports full name (0000102-monarch as well as 102)
@app.route('/<bucket>/details/<model_name>')
def model_details(bucket, model_name):
    model, model_stats = cloudy.load_model(bucket, model_name)
    if model is None:
        return 'Model {} not found'.format(model_name)
    model_id = model[0]
    model_name = model[2]
    model_num = model[5]
    run_data = cloudy.get_run_data(bucket)

    game_names = cache.get(model_name)
    if game_names is None:
        game_names = cloudy.some_model_games(
            bucket, model_id, limit=MAX_GAMES_ON_PAGE)
        game_names = sorted(list(map(os.path.basename, game_names)))

        always_include = game_names[:2] + game_names[-2:]
        if RANDOMIZE_GAMES:
            random.shuffle(game_names)

        # always_include might be useful for debugging. To avoid confusion the
        # four games are placed out of the way at the end of the list
        game_names = game_names[:MAX_GAMES_ON_PAGE-4] + always_include

        # Low cache time so that games randomize if you refresh
        cache.set(model_name, game_names, timeout=60)

    games = cloudy.load_games(bucket, game_names)

    #### MIN UNLUCK ####
    unluck_by = [
        ('black', 'unluckiness_black'),
        ('white', 'unluckiness_white'),
        ('black+white', 'unluckiness_black + unluckiness_white'),
    ]

    min_unluck = []
    for perspective, order_by in unluck_by:
        min_unluck_game = cloudy.query_db(
            'SELECT filename, {} FROM games '
            'WHERE model_id = ? AND num_moves > 70 '
            'ORDER BY 2 ASC LIMIT 1'.format(order_by),
            (model_id,))
        if min_unluck_game:
            min_unluck.append((perspective,) + min_unluck_game[0])

    opening_sgf = ''
    if model_stats is not None:
        opening_sgf = model_stats[0][14]

    policy_sgf = cloudy.get_position_eval(bucket, model_id, 'policy', 'empty')

    return render_template('model.html',
                           bucket=bucket,
                           run_data=run_data,
                           model=model,
                           model_stats=model_stats,
                           games=games,
                           min_unluck=min_unluck,
                           is_random=RANDOMIZE_GAMES,
                           opening_sgf=opening_sgf,
                           policy_sgf=policy_sgf,
                           )


@app.route('/<bucket>/graphs/<model_name>')
def model_graphs(bucket, model_name):
    # TODO: consider lookup_model_id function.
    model, model_stats = cloudy.load_model(bucket, model_name)
    if model is None:
        return 'Model {} not found'.format(model_name)
    model_id = model[0]

    # Divide by two to help avoid 'only black can win on even moves'
    game_length = cloudy.query_db(
        'SELECT black_won, 2*(num_moves/2), count(*) FROM games ' +
        'WHERE model_id = ? GROUP BY 1,2 ORDER BY 1,2', (model_id,))

    #### OPENING RESPONSES ####

    favorite_openings = cloudy.query_db(
        'SELECT SUBSTR(early_moves_canonical,'
        '              0, instr(early_moves_canonical, ";")),'
        '       count(*)'
        'FROM games WHERE model_id = ? GROUP BY 1 ORDER BY 2 DESC LIMIT 16',
        (model_id,))

    favorite_response = []
    for black_first_move, opening_count in favorite_openings:
        if not black_first_move:
            continue

        # Pass is 5 long and messes up the indexeing on 5+...
        len_move = len(black_first_move) + 2
        response = cloudy.query_db(
            'SELECT SUBSTR(early_moves_canonical, '
            '              0, ?+instr(SUBSTR(early_moves_canonical, ?, 6), ";")),'
            '       count(*) '
            'FROM games '
            'WHERE model_id = ? AND num_moves > 2 AND early_moves_canonical LIKE ?'
            'GROUP BY 1 ORDER BY 2 DESC LIMIT 8',
            (len_move, len_move, model_id, black_first_move + ';%'))

        # Trim first move
        board_size = CloudyGo.bucket_to_board_size(bucket)

        response = [(moves.split(';')[1], count) for moves, count in response]
        favorite_response.append((
            black_first_move,
            opening_count,
            sum(count for move, count in response),
            sgf_utils.commented_squares(
                board_size,
                ';B[{}]'.format(sgf_utils.cord_to_sgf(
                    board_size, black_first_move)),
                response, True, False)))

    #### SOME POLICY EVALS ####

    count, sgfs = cloudy.get_position_sgfs(bucket, [model_id])
    if sgfs:
        sgfs = sgfs[0][1:]

    return render_template('model-graphs.html',
                           bucket=bucket,
                           model=model,
                           model_stats=model_stats,
                           game_length=game_length,
                           opening_responses=favorite_response,
                           position_sgfs=sgfs,
                           )


def _embedding_serve_path(f, bucket):
    short_path = f[f.index(bucket):]
    return os.path.join("/sgf/", short_path)


@app.route('/<bucket>/nearest-neighbor/<embedding_type>')
@app.route('/<bucket>/nearest-neighbors/<embedding_type>')
def nearest_neighbor(bucket, embedding_type="value_conv"):
    file_bytes = cache.get(embedding_type)
    if file_bytes is None:
        embedding_name = 'embeddings.{}.pickle'.format(embedding_type)
        embeddings_file = os.path.join(LOCAL_EVAL_DIR, bucket, embedding_name)
        with open(embeddings_file, 'rb') as pickle_file:
            file_bytes = pickle_file.read()

        print("loaded {}, {} bytes".format(embedding_type, len(file_bytes)))
        cache.set(embedding_type, file_bytes, timeout=5 * 60)

    metadata, embeddings = pickle.loads(file_bytes)
    assert len(metadata) == len(embeddings)

    metadata = list(list(m) for m in metadata)
    for i in range(len(metadata)):
        f = metadata[i][0]
        if i == 0 and bucket not in f:
            print ("Problem!", embedding_type, f)
        f = _embedding_serve_path(f, bucket)
        metadata[i][0] = f

    x = int(request.args.get('x', 0) or 0)
    if not 0 <= x <= len(embeddings):
        return "X out of range ({})".format(len(embeddings))

    import sklearn.metrics
    distances = sklearn.metrics.pairwise.pairwise_distances(
        np.array(embeddings),
        np.array([embeddings[x]]),
        metric='l1',
        n_jobs=1).tolist()

    distances = [(d[0], i) for i, d in enumerate(distances)]
    neighbors = heapq.nsmallest(30, distances)
    neighbors = [(d, metadata[i], i) for d,i in neighbors]

    return render_template('nearest-neighbors.html',
        bucket=bucket,
        x=x,
        embed=str(embeddings[x]),
        neighbors=neighbors)


@app.route('/<bucket>/tsne/<embedding_type>')
def tsne(bucket, embedding_type="value_conv"):
    file_bytes = cache.get(embedding_type)
    if file_bytes is None:
        embedding_name = 'embeddings.{}.pickle'.format(embedding_type)
        embeddings_file = os.path.join(LOCAL_EVAL_DIR, bucket, embedding_name)
        with open(embeddings_file, 'rb') as pickle_file:
            file_bytes = pickle_file.read()

        print("loaded {}, {} bytes".format(embedding_type, len(file_bytes)))
        cache.set(embedding_type, file_bytes, timeout=5 * 60)

    metadata, embeddings, tnes = pickle.loads(file_bytes)
    assert len(metadata) == len(embeddings) == len(tnes)

    # Scale tnes to be out of ~800

    results = []
    for i in range(len(metadata)):
        # TODO: fix path hacks here: 20 for /sgf/eval/YYYY-MM-DD/'
        filename = _embedding_serve_path(metadata[i][0], bucket)[20:]
        url = url_for('eval_view', bucket=bucket, model_name=0, filename=filename)
        url += '?M=' + str(metadata[i][1])

        results.append([
            url,
            _embedding_serve_path(metadata[i][2], bucket),
            tnes[i]
        ])

    return render_template('tsne.html',
        bucket=bucket,
        results=results)

@app.route('/<bucket>/puzzles/',
            methods=['GET', 'POST'])
@app.route('/<bucket>/puzzles/<number>',
            methods=['GET', 'POST'])
def puzzles(number=0, bucket=CloudyGo.DEFAULT_BUCKET):
    puzzle_bytes = cache.get('puzzle_bytes')
    with open(os.path.join(app.static_folder, "SVM_data.json")) as SVM_data:
        puzzle_bytes = SVM_data.read()
        print("puzzle {} bytes".format(len(puzzle_bytes)))
        cache.set('puzzle_bytes', puzzle_bytes, timeout=5 * 60)

    try:
        puzzles = json.loads(puzzle_bytes)
        sgf, puzzle = puzzles[int(number)]

        # TODO(sethtroisi): Consider using original SGF and setting move num.
        sgf_path = os.path.join(
            app.instance_path, 'pro', 'problem-collection3', sgf)

        data = ''
        with open(sgf_path, 'r') as f:
            data = f.read()
        data = data.strip()
    except Exception as e:
        print (e)
        return "Did not find puzzle " + str(number)

    result_text = []
    rating_deltas = []

    show = get_bool_arg('show', request.args)
    if show or request.method == 'POST':
        assert data[-1] == ")"
        data = data[:-1]

        top_moves, coefs = puzzle
        assert len(coefs) == 10
        move_coefs = coefs[6:6+4]

        moves = []
        data += ';LB'
        for top_move, coef in zip(top_moves, move_coefs):
            j, i = divmod(top_move, 19)
            if top_move in (361, -1):
                cord = "pass"
            else:
                cord = sgf_utils.ij_to_cord(19, (i, j))

            moves.append("{:.1f} for {}".format(coef, cord))

            label = '[{}:{:.1f}]'.format(
                sgf_utils.cord_to_sgf(19, cord),
                coef)
            data += label
        data += ')'

        black_90_value = 0.8 * coefs[0] + coefs[5]
        white_90_value = -.8 * coefs[0] + coefs[1]
        even_value = coefs[3]

        value_text = "Belief in black {:.0f}, even {:.0f}, white {:.0f}".format(
            black_90_value, even_value, white_90_value)
        moves.append("other {:.1f}".format(move_coefs[-1]))
        moves_text = ", ".join(moves)

        result_text = [value_text, moves_text]
    if request.method == 'POST':
        top_moves, coefs = puzzle

        value = request.form.get('value') or 50
        move = request.form.get('move') or 'pass'

        value = float(value)
        if value > 0 and value < 1:
            value = 100 * value

        value = int(min(100, max(0, float(value))))
        result_text.append('You calculated {}% winrate for black '
                         'and would have played "{}"'.format(value, move))

        for top_move, coef in zip(top_moves, coefs[6:6+3]):
            j, i = divmod(top_move, 19)
            cord = sgf_utils.ij_to_cord(19, (i, j))
            if (cord.lower() == move.lower()):
                move_v = coef
                break
        else:
            move_v = coefs[6+3]

        value_0 = (2 * (value/100) - 1) * coefs[0]
        value_bucket = int(4.9999 * (value/100))
        value_1 = coefs[1:6][value_bucket]

        rating_deltas = [
            value,
            value_0 + value_1,
            move,
            move_v,
        ]

    return render_template('puzzle.html',
                           bucket=bucket,
                           number=number,
                           name=sgf,
                           data=data,
                           result_text=result_text,
                           rating_deltas=rating_deltas,
                           )


@app.route('/<bucket>/json/eval-pairs.json')
def eval_json(bucket):
    # Not used by CloudyGo but easy to support for external people

    model_range = CloudyGo.bucket_model_range(bucket)

    data = cloudy.query_db(
        'SELECT '
        '   model_id_1 % 1000000, model_id_2 % 1000000, '
        '   m1_black_wins + m1_white_wins, games '
        'FROM eval_models '
        'WHERE (model_id_1 BETWEEN ? AND ?) AND model_id_1 < model_id_2',
        model_range)
    return jsonify(data)


@app.route('/<bucket>/json/ratings.json')
def ratings(bucket):
    # Not used by CloudyGo but easy to support for external people

    model_range = CloudyGo.bucket_model_range(bucket)
    ratings = cloudy.query_db(
        'SELECT model_id_1 % 1000000, round(rankings, 3), round(std_err,3) '
        'FROM eval_models '
        'WHERE model_id_2 == 0 AND '
        '      model_id_1 BETWEEN ? and ?',
        model_range)
    return jsonify(ratings)

