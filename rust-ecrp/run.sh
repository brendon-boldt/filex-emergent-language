#!/bin/sh

set -ex

mkdir -p output figures

configs="alpha beta n_iters n_params"

for c in $configs; do
    cargo run --release $c > output/$c.out
    
    cd ..
    env/bin/python -m simple_nav.expectation_crp $c rust-ecrp/output/$c.out
    cd rust-ecrp
done

mv ../results/model-* figures
