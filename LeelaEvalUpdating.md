# Steps for running and updating leela eval

```
# Get new model list
cd ~/Projects/cloudygo
oneoff/leela-model-importer.py
oneoff/leela-model-importer.py leela-zero-eval

# Update sql db
sqlite3 instance/clouds.db ".mode csv" ".import instance/data/leela-zero/inserts.csv models"
sqlite3 instance/clouds.db ".mode csv" ".import instance/data/leela-zero-eval/inserts.csv models"

# Remove empty weight files from leela-model-importer.py
find instance/data/leela-zero/models/ -iname 'LZ*' -type f -empty
find instance/data/leela-zero/models/ -iname 'LZ*' -type f -empty -delete

# Download good weight files
cd instance/data/leela-zero
./download.sh

# Rsync weights to fast (this is Seth specific)
rsync -tr --info=progress2 "models/LZ*" eights@fast.cloudygo.com:~/Projects/ringmaster/leela-zero-v3/

# Later sync games back with this helpful macro
cd ~/Projects/cloudygo
function sync_and_update_lz3 { rsync -tr --info=progress2 "eights@fast.cloudygo.com:~/Projects/ringmaster/leela-zero-v3-eval.games" instance/data/leela-zero-eval/eval/; }

# Update cloudygo
sync_and_update_lz3; ./updater.py eval_games leela-zero-eval leela-zero-eval-time
```
