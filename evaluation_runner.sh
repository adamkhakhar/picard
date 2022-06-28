for num_predictions in 1 2 3 4 5 6 7 8 9 10 20 50 100
do
    echo num_predictions $num_predictions
    python3 code-prediction-set/config_update.py $((num_predictions+4)) $num_predictions
    for startindx in {0..200..10}
    do
        make serve --ignore-errors >> log &
        echo startindx $startindx
        sleep 30
        python3 code-prediction-set/runner.py $((num_predictions+4)) $num_predictions --startindx $startindx --endindx $((startindx +10))
        podman kill `podman ps -q`
    done
done