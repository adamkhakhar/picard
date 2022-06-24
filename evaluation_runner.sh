for startindx in {0..0..30}
do
    make serve >> log &
    sleep 30
    python3 code-prediction-set/runner.py 1 --startindx $startindx --endindx $((startindx +31))
    podman kill `podman ps -q`
done