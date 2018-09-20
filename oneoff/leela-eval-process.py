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
import sys

from tqdm import tqdm

#assert len(sys.argv) == 2, "Must pass bucket"

#bucket = os.path.join("instance", "data", sys.argv[1])
bucket = "/home/eights/test/lz-eval"
assert os.path.exists(bucket), bucket

os.chdir(bucket)

shorten_hash = re.compile(r"(\[[0-9a-f]{8})[0-9a-f]{56}")
leela_strip = re.compile(r"\[Leela\s*Zero\s*([0-9](\.[0-9]+)+)?\s+(networks)?\s*")
name_extractor = re.compile(r"PB\[(.*)\]PW\[([^]]*)\]")
move_notation = re.compile(r"([^:]*):_?([0-9a-f]{8})_([0-9a-f]{8})")
# network_dedup="s#\([0-9a-f]\{8\}\)\(_\1\)\?#\1#"

valid_networks = os.listdir(os.path.join(bucket, "models"))
short_nets = set(n[:8] for n in valid_networks)
print("{} networks".format(len(short_nets)))

os.chdir('eval')
for n in short_nets | {'../nonprod', '../unknown'}:
    try:
        os.mkdir(n)
    except FileExistsError:
        pass

with open("../versions") as versions, \
     open("../raw_moves", "w") as raw_moves, \
     open("../nonprod_moves", "w") as nonprod_moves:
    for players in tqdm(versions):
        # Shorten hash
        players = players.strip()
        original = players
        players = shorten_hash.sub(r"\1", players, re.I)
        players = leela_strip.sub("[", players, re.I)
        names = name_extractor.sub(r"\1_\2", players, re.I)
        move_parts, n =  move_notation.subn(r"\1 \2 \3", names, re.I)
        if n == 1:
            f, n1, n2 = move_parts.split(" ")
            move = "{} {}_{}\n".format(f, n1, n2)
            if n1 in short_nets and n2 in short_nets:
                raw_moves.write(move)
                os.rename(f, os.path.join(n1, f))
            else:
                nonprod_moves.write(move)
                os.rename(f, os.path.join('../nonprod', f))
        else:
            # these files still need to be moved somewhere
            f = original.split(':')[0]
            os.rename(f, os.path.join('../unknown', f))

            if "Human" in original or "networks]" in original:
                continue
            print (original)
            print (players)
            print (names, move_parts, n)
            print ()

