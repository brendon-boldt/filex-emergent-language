#!/bin/sh

set -xe

cfgs="learning_rate buffer_size lexicon_size train_steps"
n_threads=4

for c in $cfgs; do
    python -m simple_nav run $c -j$n_threads
done

for c in $cfgs; do
    python -m simple_nav eval log/$c -j$n_threads
done

for c in $cfgs; do
    python -m simple_nav analyze $c
done
