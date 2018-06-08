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

import re
import shlex
import sys
import os
from collections import defaultdict
from subprocess import Popen, check_output, PIPE, STDOUT

timestamp = "000000" if len(sys.argv) < 2 else sys.argv[1]
offset = 0 if len(sys.argv) < 3 else int(sys.argv[2])

# renames import/[0-9]*_[0-9]*.sgf to B_W_<num+offset>

# TODO how to derive imperically (i.e. import cloudygo)
# bucket_hash=15
# ls import/ | cut -d- -f2 | sort -u | sed 's/^0*//' | xargs -I {} sh -c 'echo "$(('$bucket_hash' * 1000000 + {})),{},{},'$bucket',{},0,0,0,0,0,0' | sort -n | tee inserts.csv

p1 = Popen(["find", "import", "-iname", "*sgf"], stdout=PIPE, stderr=STDOUT)
players = check_output(
    shlex.split('xargs grep -o "\(Black\|White\) \([0-9]\{6\}\)-"'),
    stdin=p1.stdout,
    stderr=PIPE)

lines = players.decode('utf-8').strip().split("\n")

file_parts = defaultdict(lambda: ["", ""])
for line in lines:
    parts = re.split(r"[ :]+", line)
    filename, color, net = parts
    assert color in ("Black", "White")
    assert net.endswith('-')

    file_parts[parts[0]][0 if color == "Black" else 1] = net[:-1]

assert 2 * len(file_parts) == len(lines)

for f, (b, w) in file_parts.items():
    name = os.path.basename(f)
    dirname = os.path.dirname(f)
    matchup, game = map(int, name.replace(".sgf", "").split("_"))

    game_num = game + offset

    new_name = "-".join([timestamp, w, b, str(game_num)]) + ".sgf"
    new_path = os.path.join(dirname, new_name)
    os.rename(f, new_path)

print(".mode csv")
print(".import instance/data/$bucket/inserts.csv models")
