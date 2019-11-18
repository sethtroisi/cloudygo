# Downloaded
`all_match.sgf.xz`
`all.sgf.xz`

```sh
./oneoff/delete_bucket.py leela-zero
./oneoff/setup-run.sh leela-zero
```

```sh
./oneoff/leela-model-importer.py leela-zero
sqlite3 instance/clouds.db ".mode csv" ".import instance/data/leela-zero/inserts.csv models"
```

```sh
./oneoff/leela-all-to-dirs.sh
```

TODO hashing games for consistent urls or redo each game

## Import into cloudygo!
```sh
./updater.py models leela-zero
./updater.py eval_games leela-zero
./updater.py games leela-zero
```


# Eval Process
```sh
cd instance/data/leela-zero
# (manually delete LZXXX_0123ABDC) for new models)
./download.sh
```

# CloudyGo.com specific
```sh
rsync -tr --info=progress2 "~/Projects/cloudygo/instance/data/leela-zero/models/LZ*" $OTHER_PC:~/Projects/ringmaster/leela-zero-v3/
ssh $OTHER_PC
cd Projects/ringmaster/
screen
export CTL=leela-zero-v3-eval.ctl; date; (ls "${CTL%.ctl}.games/" | wc -l); ringmaster "$CTL" show | grep -o '[0-9]*/\?[0-9]* games' | sort | uniq -c
time ringmaster leela-zero-v3-eval.ctl run
```

```sh
function sync_and_update_lz3 { rsync --info=progress2 -t -r "$OTHER_PC:~/Projects/ringmaster/leela-zero-v3-eval.games" ~/Projects/cloudygo/instance/data/leela-zero-eval/eval/; }
sync_and_update_lz3
```
