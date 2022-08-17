
for num_predictions in 8
do
    beam_size=16
    mkdir -p /home/akhakhar/code/picard/code-prediction-set/results_v2/beam_size_$beam_size/num_predictions_$num_predictions
    index_step_size=5
    num_samples=200
    echo num_predictions $num_predictions beam_size $beam_size
    python3 code-prediction-set/config_update.py $beam_size $num_predictions
    for ((startindx = 0; startindx <= $((num_samples - 1)); startindx += $index_step_size))
    do
        make serve --ignore-errors >> log &
        echo startindx $startindx
        sleep 90
        python3 code-prediction-set/runner.py $beam_size $num_predictions --startindx $startindx --endindx $((startindx + $index_step_size)) --ignore_rep 0
        podman cp `podman ps -q`:/app/seq2seq/tmp_cache.bin /home/akhakhar/code/picard/code-prediction-set/results_v2/beam_size_$beam_size/num_predictions_$num_predictions/startindx_$startindx
        podman kill `podman ps -q`
    done
done