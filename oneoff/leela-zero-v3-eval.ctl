stderr_to_log = True

import os.path
import glob

BASE = "leela-zero-v3/"

def LeelaPlayer(model, playouts):
    # Full strength LZ
    return Player(
        "./leelaz -q -g --noponder -w {} -t 1 -v {}".format(
            model, playouts),
        override_name=True)

matchups = []
def add_matchup(matchup_id, model_a, model_b, games):
    matchups.append(Matchup(
        model_a, model_b,
        id=matchup_id, number_of_games=games,
        alternating=True, scorer='players'))

visits = [
    ('_rapid', 50),
    ('_fast', 200),
    ('_slow', 800),
]

networks = {}
players = {}
for filename in sorted(glob.glob(BASE + "*")):
    model = os.path.basename(filename)
    number = int(model.split('_')[0][2:])
    networks[number] = model

    for visits_name, visits_count in visits:
      players[model + visits_name] = LeelaPlayer(filename, visits_count)

net_diffs = [1,2,3,5,10,15,20,25,50]
for m_i, model_a in sorted(networks.items()):
    for d in net_diffs:
        m_j = m_i - d
        model_b = networks.get(m_j)
        if model_b is not None:
            games = 10 + 6 * (m_j >= 180) + 4 * (m_j <= 30)

            matchup_name = 'LZ{}_vs_LZ{}'.format(m_i, m_j)
            for visit_name, _ in visits:
                add_matchup(
                    matchup_name + visit_name,
                    model_a + visit_name,
                    model_b + visit_name,
                    games + (4 if visit_name != '_slow' else 0))

