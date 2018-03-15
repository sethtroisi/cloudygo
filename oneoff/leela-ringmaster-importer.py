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
offset    = 0        if len(sys.argv) < 3 else int(sys.argv[2])

# Runs from leelaz/v1/ directory and renames import/ games to B_W_<num+offset>


p1 = Popen(["find", "import", "-iname", "*sgf"], stdout=PIPE, stderr=STDOUT)
players = check_output(
    shlex.split('xargs grep -o "\(Black\|White\) \([0-9a-f]\{8\}\(_fast\|_slow\)\) Leela Zero"'),
    stdin=p1.stdout,
    stderr=PIPE)


lines = players.strip().split("\n")
print (len(lines))

file_parts = defaultdict(lambda: ["", ""])
for line in lines:
    parts = re.split(r"[ :]+", line)
    filename, color, net, leela, zero = parts
    assert leela == "Leela" and zero == "Zero", parts
    assert color in ("Black", "White")

    file_parts[parts[0]][0 if color == "Black" else 1] = net

assert 2 * len(file_parts) == len(lines)

for f, (b, w) in file_parts.items():
    name = os.path.basename(f)
    dirname = os.path.dirname(f)
    matchup, game = map(int, name.replace(".sgf", "").split("_"))

    slow_offset = 10 * (w+b).count("_slow")
    game_num = game + offset + slow_offset

    new_name = "-".join([timestamp, w, b, str(game_num)]) + ".sgf"
    new_path = os.path.join(dirname, new_name)
    os.rename(f, new_path)

# rename 's#000-([0-9]*)-([0-9]*)#001-$2-$1#' *.sgf

