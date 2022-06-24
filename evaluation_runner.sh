for startindx in {0..195..15}
do
    make serve --ignore-errors >> log &
    sleep 30
    python3 code-prediction-set/runner.py 1 --startindx $startindx --endindx $((startindx +21))
    podman kill `podman ps -q`
done