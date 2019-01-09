echo "all-eval cron $(date)"

set -e

cd instance/data/synced-eval/

mkdir -p eval/cross-eval \
         eval/v9-19x19 eval/v10-19x19 \
         eval/v12-19x19 eval/v13-19x19 \
         eval/v14-19x19 eval/v15-19x19

eval_rsync="rsync -ah --info=progress2 --ignore-existing"

#$eval_rsync "../v9-19x19/eval/"  "eval/v9-19x19"
#$eval_rsync "../v10-19x19/eval/" "eval/v10-19x19"
#$eval_rsync "../v12-19x19/eval/" "eval/v12-19x19"
#$eval_rsync "../v13-19x19/eval/" "eval/v13-19x19"
#$eval_rsync "../v14-19x19/eval/" "eval/v14-19x19"
$eval_rsync "../v15-19x19/eval/" "eval/v15-19x19"

# MAKE LIST OF INTERESTING cross-eval files
cross_bucket_regex='.*-\(v9\|v10\|v1[2-9]\)-.*-\(v9\|v10\|v1[2-9]\)-.*'
find ../cross-eval/eval/ -regextype sed -regex "$cross_bucket_regex" \
    | sed 's#../cross-eval/eval/##' | tqdm > cross_games.txt

$eval_rsync --files-from cross_games.txt "../cross-eval/eval/" "eval/cross-eval"

echo "all-eval update $(date)"
./updater.py eval_games synced-eval
