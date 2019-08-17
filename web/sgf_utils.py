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

from subprocess import Popen, PIPE, STDOUT

import math
import re
import os.path


def chunk(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


def cord_to_ij(board_size, move):
    if move == 'pass':
        return board_size, 0

    i = ord(move[0]) - ord('A')
    return i - (i > 8), board_size - int(move[1:])


def ij_to_cord(board_size, ij):
    if ij == (board_size, 0):
        return 'pass'

    return cord_name(ij[0]) + str(board_size - ij[1])


def sgf_name(i):
    return chr(ord('a') + i)


def cord_to_sgf(board_size, cord):
    if 'pass' in cord:
        return ''

    index1 = ord(cord[0].upper()) - ord('A')
    if index1 > 8:
        index1 -= 1

    index2 = int(cord[1:])

    return sgf_name(index1) + sgf_name(board_size - index2)


def cord_name(i):
    return chr(ord('A') + i + (i >= 8))


def sgf_to_cord(board_size, move):
    if len(move) == 3 or move.endswith('[tt]'):
        return 'pass'

    assert len(move) == 5, move  # expects B[dc]
    index1 = ord(move[2]) - ord('a')
    index2 = ord(move[3]) - ord('a')

    assert 0 <= index1 < board_size, index1
    assert 0 <= index2 < board_size, index2

    return cord_name(index1) + str(board_size - index2)


def count_moves(sgf):
    return sgf.count(';B[') + sgf.count(';W[')


def pretty_print_sgf(data):
    sep = '<br>'

    LINE_LEN = 100
    return sep.join(
        sep.join(line[i:i+LINE_LEN] for i in range(0, len(line), LINE_LEN))
        for line in data.split('\n'))


def commented_squares(board_size, setup, data, include_move, is_pv):
    color = 'W' if setup.rfind(';B') > setup.rfind(';W') else 'B'

    squares = ''
    if include_move:
        c = color
        for i, move in enumerate(data):
            squares += ';{}[{}]'.format(c, cord_to_sgf(board_size, move[0]))
            if is_pv:
                c = 'W' if c == 'B' else 'B'

    def label(d):
        comment_fmt = "{:g}" if isinstance(d[1], float) else "{}"
        comment = comment_fmt.format(d[1])
        return '[{}:{}]'.format(cord_to_sgf(board_size, d[0]), comment)

    labels = '' if is_pv else ';LB' + ''.join(map(label, data))
    return '(;DT[2019]SZ[{}]KM[7.5]{}{}{})'.format(
        board_size, setup, squares, labels)


def board_png(board_size, setup, data, filename=None,
              include_move=True, is_pv=False, force_refresh=False):
    sgf = commented_squares(board_size, setup, data, include_move, is_pv)
    if filename:
        if force_refresh or not os.path.exists(filename):
            try:
                p = Popen(['sgftopng', filename, '-coord', '-nonrs'],
                          stdout=PIPE, stdin=PIPE, stderr=STDOUT)
                sgf_to_png_stdout = p.communicate(input=sgf.encode('utf-8'))[0]
            except FileNotFoundError:
                # sgftopng not found.
                # model page and model-graphs won't have pretty images.
                pass
    return sgf


def rotate(board_size, ij, rot):
    if ij == (board_size, 0):
        # pass is considered oriented I guess
        return ij

    i, j = ij

    # vertical flip
    if rot & 1:
        i, j = board_size - 1 - i, j
    # rotate counterclockwise
    if rot & 2:
        i, j = board_size - 1 - j, i
    # rotate 180
    if rot & 4:
        i, j = board_size - 1 - i, board_size - 1 - j
    return i, j


def canonical_rotation(board_size, moves):
    if len(moves) == 0:
        return 0

    # First move should be in [K-T][10-19] under the diagonal
    ij = cord_to_ij(board_size, moves[0])
    if moves[0] == 'pass' or (ij[0] == ij[1] and 2 * ij[0] + 1 == board_size):
        # Pass or tengen, canonicalie other moves
        return canonical_rotation(board_size, moves[1:])

    test_rots = []
    for rot in range(8):
        i, j = rotate(board_size, ij, rot)
        if 2*i >= board_size-1 and 2*j <= board_size-1 and i + j + 1 >= board_size:
            if i + j + 1 == board_size:
                # stone on diagonal, check next stone
                test_rots.append(rot)
            elif 2*j + 1 == board_size:
                # stone on center axis
                test_rots.append(rot)
            else:
                # inner part of quadrant
                return rot

    assert len(test_rots) == 2, (test_rots, moves)
    if len(moves) == 1:
        return test_rots[0]

    # Find first stone differentiated by test_rots
    # Return rotation with lower j or fallback lower i
    for move in moves[1:]:
        ij = cord_to_ij(board_size, move)
        for r, rot in enumerate(test_rots):
            test = rotate(board_size, ij, rot)
            if ij != test:
                other_rot = test_rots[1 - r]

                if test[1] != ij[1]:
                    return rot if test[1] < ij[1] else other_rot
                return rot if test[0] < ij[0] else other_rot

    #print ("Canonical_rotation confustion", test_rots, moves)
    return test_rots[0]


def canonical_moves(board_size, moves):
    # Takes moves in cord (e.g. D13) form.
    move_list = moves.split(';')
    if len(moves) == 0:
        return moves

    rot = canonical_rotation(board_size, move_list)

    def rotated(m):
        return ij_to_cord(
            board_size,
            rotate(
                board_size,
                cord_to_ij(board_size, m),
                rot))

    return ';'.join(map(rotated, move_list))


def canonical_sgf(board_size, sgf):
    if not sgf:
        return sgf

    # NOTE: This should really utilize a real SGF parser...
    # One with tests and better utils...
    tokens = list(re.finditer('(;[BW]\[(..)\]|\[(..):)', sgf))
    moves = [token.group(2) or token.group(3) for token in tokens]

    # Silly but what you doing to do.
    cords = ';'.join([sgf_to_cord(board_size, 'B[' + m + ']') for m in moves])
    canonical = canonical_moves(board_size, cords).split(';')
    new_moves = [cord_to_sgf(board_size, c) for c in canonical]

    new_sgf = list(sgf)
    for token, move, new_move in zip(tokens, moves, new_moves):
        # If you change this test it PLEASE
        new_token = list(token.group(0).replace(move, new_move))
        new_sgf[token.start():token.end()] = new_token
        #print (token.span(), move, new_move, "\t", ''.join(new_sgf))

    return ''.join(new_sgf)


def read_game_data(game_path):
    try:
        with open(game_path, 'r') as f:
            return f.read()
    except:
        print('Unable to parse:', game_path)
    return None


def fully_parse_comment(comment):
    # SLOW!

    assert comment.startswith('C['), comment
    assert comment.endswith(']'), comment

    tokens = re.split(r'[\n :,]+', comment[2:-1])

    # KataGo format is float, float, float, float
    if len(comment) in range(21, 24+1):
        return None

    # comment format is:
    # <OPTIONAL resign rate>
    # <OPTIONAL model_name>
    # <Q root>\n
    # PV_move_1 (visits_1) ==> PV_move_2 (visits_2) ... ==> Q:<Q>\n
    # move: action Q U P P-Dir N soft-N p-delta p-rel
    # <15 rows>

    if len(tokens) > 5 and tokens[0] == 'move' and tokens[2] in 'BW':
        # LEELA-HACK for ringmaster eval
        assert tokens[8][-1] == '%'
        playouts = tokens[6]
        winrate = float(tokens[8][:-1]) / 100
        Q_0 = 2 * winrate - 1
        # Winrate is from player to play, not from black
        if tokens[2] == 'W':
            Q_0 *= -1

        pv_moves = tokens[10:]
        return ("LZ", 0), (pv_moves, [playouts]), (Q_0, Q_0), []

    resign = None
    if tokens[0] == 'Resign':
        resign = float(tokens[2])
        tokens = tokens[3:]

    model = ""
    if not re.match(r'^-?[01].[0-9]*$', tokens[0]):
        # MINIGO-HACK for https://github.com/tensorflow/minigo/issues/652
        if re.match(r'^[0-9]{6}-[a-z-]*(.pb)?$', tokens[0]):
            model = tokens.pop(0)

        # Assume it's model name and drop for now
        if tokens[0] == 'models':
            raw_model = tokens.pop(0)
            while tokens[0] == 'gs':
                temp = tokens.pop(0)
                temp = tokens.pop(0)
                model = os.path.basename(temp)
                assert model.startswith('model.ckpt'), model

            if tokens[0].endswith('.pb'):
                model = tokens.pop(0)

    Q_0 = float(tokens.pop(0))

    pv_count = tokens.count('==>')
    pv_raw = tokens[:3*pv_count]
    tokens = tokens[3*pv_count:]

    pv_moves = pv_raw[0::3]
    pv_counts = [int(count[1:-1]) for count in pv_raw[1::3]]

    q_header = tokens.pop(0)
    assert q_header == 'Q', '{}, {}'.format(q_header, tokens)
    Q_PV = float(tokens.pop(0))

    # return a minified table of things we use
    header_len = 10
    table = [[tokens[0], tokens[6], tokens[7]]]
    for row in range(header_len, len(tokens), header_len):
        table.append((tokens[row+0], int(tokens[row+6]), float(tokens[row+7])))

    #table = list(chunk(tokens, header_len))
    # for row in table[1:]:
    #    for i, v in enumerate(row[1:], 1):
    #        row[i] = float(v)

    return (model, resign), (pv_moves, pv_counts), (Q_0, Q_PV), table


def derive_move_quality(played_moves, parsed_comments):
    # TODO(sethtroisi): VERIFY THIS WORKS FOR PASS

    # for each move measure how many visits were to nodes above it
    unluckness = []
    visits = []

    for move_num, (played, comment_data) in enumerate(zip(played_moves, parsed_comments)):
        sum_soft_n_better = 0
        visits_played = 0
        visit_index = comment_data[3][0].index('N')
        for row in comment_data[3][1:]:
            if row[0] == played:
                visits_played = row[visit_index]
                break
            sum_soft_n_better += row[visit_index+1]
        else:
            # TODO(sethtroisi): discuss if frequency here is bad
            # print ('Missing {} move saw {} of soft-n'.format(
            #    played, sum_soft_n_better))
            pass

        # seems to happen with equal visit count
        # if move_num > 30 and sum_soft_n_better > 0:
        #    print(move_num, played, [row[0] for row in comment_data[3][1:]])

        all_visits_seen = sum([row[visit_index]
                               for row in comment_data[3][1:]])
        #print (played, visits_played, sum_soft_n_better, all_visits_seen)

        # TODO(sethtroisi): if not found, maybe add half of remaining soft-n

        unluckness.append(min(sum_soft_n_better, 1.0))
        visits.append(visits_played)

    return visits, unluckness


def parse_game_simple(game_path, data=None, include_players=False):
    if data is None:
        data = read_game_data(game_path)
        if data is None:
            return None

    match = re.search(r'RE\[([wWbB][^]]+)\]', data)
    if not match:
        # This is a known issue, ignore these games
        if 'RE[None]' in data:
            return None
        # This is present in ~3k KataGo games
        if 'RE[0]' in data:
            return None

    assert match, game_path

    result = match.group(1).upper()
    black_won = 'B' in result

    moves = data.count(';') - 1

    if include_players:
        PB = re.search(r'PB\[([^\]]*)\]', data)
        PB = PB.group(1) if PB else ""

        PW = re.search(r'PW\[([^\]]*)\]', data)
        PW = PW.group(1) if PW else ""
        return black_won, result, moves, PB, PW

    return black_won, result, moves


def raw_game_data(filepath, data):
    # TODO this doesn't find comments not on moves.
    # UGLY HACK to allow comment before or after W[] B[] tag.
    raw_moves = list(re.finditer(
        r';\s*([BW]\[[a-t]*\]|C\[[^]]*\])\s*([BW]\[[a-s]*\]|C\[[^]]*\])?',
        data))

    moves = []
    comments = []
    for match in raw_moves:
        if match.group(1).startswith('C'):
            comments.append(match.group(1))
            moves.append(match.group(2))
        else:
            moves.append(match.group(1))
            if match.group(2):
                comments.append(match.group(2))

    # format is: resign, (pv_moves, pv_counts), (Q0, Qpv), table
    parsed_comments = list(map(fully_parse_comment, comments))
    return moves, parsed_comments


PLAYER_BLACK_RE = re.compile(r'PB\[([^]]*)\]')

def parse_game(game_path):
    data = read_game_data(game_path)
    if data is None: return None

    board_size = 9 if 'SZ[9]' in data else 19

    # Note: PB and PW are not requested here
    result = parse_game_simple(game_path, data)
    if not result: return None
    black_won, result, num_moves = result

    result_margin = float(result.split('+')[1]) if ('+R' not in result) else 0

    moves, parsed_comments = raw_game_data(game_path, data)

    # LEELA-HACK for getting model
    # TODO: verify this hack with LZ people.
    if ('leela' in game_path) or ('KataGo' in game_path):
        match = PLAYER_BLACK_RE.search(data)
        if match:
            model = match.group(1)

        parsed_comments = []
    else:
        # TODO this was broken by Tom.
        # TODO LZ needs to look for -w ..... and parse that
        model = parsed_comments[0][0][0] if parsed_comments else ""

    played_moves = [sgf_to_cord(board_size, move) for move in moves]

    first_two_moves = ';'.join(played_moves[:2])
    early_moves = ';'.join(played_moves[:10])
    early_moves_canonical = canonical_moves(board_size, early_moves)

    move_quality = derive_move_quality(played_moves, parsed_comments)

    unluckiness_black = round(sum(move_quality[1][:30:2]), 3)
    unluckiness_white = round(sum(move_quality[1][1:30:2]), 3)

    # Sometimes move with equal N in 2nd position is choosen
    #assert sum(move_quality[1][30:]) == 0, game_path

    has_stats = len(parsed_comments) > 0

    top_move_visit_count = [parsed[1][1][0] for parsed in parsed_comments]
    number_of_visits_b = sum(top_move_visit_count[0::2])
    number_of_visits_w = sum(top_move_visit_count[1::2])
    number_of_visits_early_b = sum(
        count for count in top_move_visit_count[0:30:2])
    number_of_visits_early_w = sum(
        count for count in top_move_visit_count[1:30:2])

    if has_stats and len(parsed_comments) > 2:
        resign_threshold = parsed_comments[0][0][1]
        assert resign_threshold is not None

        bleakest_eval_black = min(parsed[2][0]
                                  for parsed in parsed_comments[0::2])
        bleakest_eval_white = max(parsed[2][0]
                                  for parsed in parsed_comments[1::2])
    else:
        # LEELA-HACK for finding resign rate.
        resign_rate = re.findall(r' -r ([0-9]*) ', data)
        assert len(resign_rate) <= 1, (game_path, resign_rate)
        if resign_rate:
            # Otherwise "holdout_resign" assert gets thrown
            resign_threshold = -0.999 + 0.01 * int(resign_rate[0])
        else:
            resign_threshold = -0.99

        bleakest_eval_black = 1 if black_won else -1
        bleakest_eval_white = -1 if black_won else 1

    if (bleakest_eval_black is None or bleakest_eval_white is None or
       math.isnan(bleakest_eval_black) or math.isnan(bleakest_eval_white)):
        print("Bad bleakest:", game_path)
        bleakest_eval_black = 1 if black_won else -1
        bleakest_eval_white = -1 if black_won else 1

    return (model,
            black_won, result, result_margin,
            num_moves,
            first_two_moves, # NOTE: used to be early_moves
            early_moves_canonical,
            has_stats,
            number_of_visits_b, number_of_visits_w,
            number_of_visits_early_b, number_of_visits_early_w,
            unluckiness_black, unluckiness_white,
            resign_threshold, bleakest_eval_black, bleakest_eval_white)
