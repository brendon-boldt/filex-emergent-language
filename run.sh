#!/bin/sh

# Switch to 1 if debugging code
n_threads=$(nproc)

# Space delimited list of configuration found in simple_nav/experiment_configs.py and simple_nav/analysis_configs.py

for env in nodyn recon sig nav; do
    for ind_var in learning_rate buffer_size lexicon_size timesteps temperature; do
        cfgs="$cfgs ${env}_$ind_var"
    done
done

set -xe

# Test that the environment is set up correct with the following
# cfgs="quick_test"

for c in $cfgs; do
    # Train the models; output is put in log/config_name
    python -m simple_nav run $c -j$n_threads
done

for c in $cfgs; do
    # Evaluate the models; output is put in results/config_name
    python -m simple_nav eval log/$c -j$n_threads
done

for c in $cfgs; do
    # Analyze the eval data; output is put in results/config_name
    python -m simple_nav analyze $c
done
