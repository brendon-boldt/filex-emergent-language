#!/bin/sh

set -e

mkdir -p output figures

configs="alpha beta n_iters n_params"

for c in $configs; do
    echo Runing $c...
    # The code is not optimized (i.e., slow) without `--release`
    cargo run --release $c > output/$c.out
    
    cd ..
    # We load the data into matplotlib to make everything look consistent
    env/bin/python -m simple_nav.expectation_crp $c model-rust/output/$c.out
    cd model-rust
done

mv ../results/model-* figures
