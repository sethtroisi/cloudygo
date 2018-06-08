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

import math
import os
import random
import re
import sqlite3
import time

from collections import defaultdict

from werkzeug.contrib.cache import SimpleCache
from flask import Flask, g
from flask import request, render_template
from flask import send_file, url_for, redirect


from . import sgf_utils
from .cloudygo import CloudyGo

app = Flask(__name__)
app.jinja_env.trim_blocks = True
app.jinja_env.lstrip_blocks = True

cache = SimpleCache()

LOCAL_DATA_DIR = os.path.join(app.instance_path, 'data')
DATABASE_PATH = os.path.join(app.instance_path, 'clouds.db')

RANDOMIZE_GAMES = True
MAX_GAMES_ON_PAGE = 100
MAX_INSERTS = 5000

#### DB STUFF ####

# def get_db():
#    db = getattr(g, '_database', None)
#    if db is None:
#        db = g._database = sqlite3.connect(DATABASE_PATH)
#        db.row_factory = sqlite3.Row
#    return db


#@app.teardown_appcontext
# def close_connection(exception):
#    db = getattr(g, '_database', None)
#    if db is not None:
#        db.close()


#print('Setting up Cloudy')
# cloudy = CloudyGo(
#    app.instance_path,
#    LOCAL_DATA_DIR,
#    get_db,
#    cache,
#    None, # multiprocessing pool
#)

#### UTILS ####

def get_bool_arg(name, args):
    value = args.get(name, 'false').lower()
    return value not in ('f', 'false')

#### PAGES ####


@app.route('/<bucket>/play-cloudygo', methods=['GET'])
def play_cloudygo(bucket):
    model_name = cloudy.get_newest_model_num(bucket)

    # TODO: Figure out where this code and logic should live
    current_game = os.path.join(
        app.instance_path, 'play-cloudygo', 'current.sgf')

    # TODO: Deal with game finished, new game, ...
    with open(current_game) as data:
        game_sgf = data.read().replace('\n', '')
    minigo_name = 'Minigo {} {}'.format(bucket, model_name)
    game_sgf = game_sgf.replace('PB[]', 'PB[' + minigo_name + ']')
    game_sgf = game_sgf.replace('PW[]', 'PW[Humans with help from Minigo]')

    # Play a move if it's been enough time
    current_votes = cache.get('play-cloudygo-game') or {}
    now = time.time()
    if now > (cache.get('play-cloudygo-countdown') or now) or \
       now > (cache.get('play-cloudygo-vote-last') or now):
        # Make this syncrounaoues

        # Play top voted move
        assert len(current_votes) > 0
        top_move, votes = max(current_votes.items(), key=lambda kv: kv[1])
        # TODO log this or something
        print(top_move, 'with', votes, 'votes')
        # TODO deal with pass / resign

        if top_move == 'pass':
            move = ''
        elif top_move == 'resign':
            # TODO
            move = ''
        else:
            move = sgf_utils.cord_to_sgf(top_move)

        to_play = 'W' if game_sgf.rfind(';B') > game_sgf.rfind(';W') else 'B'
        game_sgf = '{};{}[{}])'.format(game_sgf[:-1], to_play, move)

        # TODO please no more bad code

        with open(current_game, 'w') as temp:
            print('writing:', game_sgf)
            temp.write(game_sgf + '\n')

        # Clear all current move state
        cache.delete('play-cloudygo-countdown')
        cache.delete('play-cloudygo-vote-last')
        cache.delete('play-cloudygo-game')
        current_votes = {}

    poll_options = []
    for i in range(19*19):
        move = sgf_utils.ij_to_cord(divmod(i, 19))

        votes = current_votes.get(move, 0)
        policy = 0

        label = []
        if votes > 0:
            label.append(str(votes) + 'x')
        if policy > 0.01:
            label.append('P: {:.0f}'.format(100 * policy))

        poll_options.append((move, ' '.join(label)))
    poll_options.append(('pass', 'pass'))
    poll_options.append(('resign', 'resign'))

    # TODO history of games

    return render_template('play-cloudygo.html',
                           bucket=bucket,
                           model_name=model_name,
                           game_sgf=game_sgf,
                           poll_options=poll_options,
                           )


@app.route('/<bucket>/play-cloudygo-submit', methods=['POST'])
def vote_play_cloudygo(bucket):
    move_vote = request.values.get('move', None)
    if move_vote == None:
        return play_cloudygo(bucket)

    key = 'vote from {}'.format(request.remote_addr)
    voted_for = cache.get(key)
    if voted_for:
        return 'Voted recently (for {}), wait ~5s'.format(voted_for)
    else:
        cache.set(key, move_vote, timeout=6)

    # Save this in an atomic synconous thread safe data structure
    cache.add('play-cloudygo-game', {}, timeout=0)
    current_votes = cache.get('play-cloudygo-game')
    votes = current_votes.get(move_vote, 0)
    current_votes[move_vote] = votes + 1
    cache.set('play-cloudygo-game', current_votes)

    # Wait in increments of 5 up to 30
    cache.add('play-cloudygo-countdown', time.time() + 30, timeout=0)
    cache.set('play-cloudygo-vote-last', time.time() + 5, timeout=0)

    return redirect(url_for('.play_cloudygo', bucket=bucket))
