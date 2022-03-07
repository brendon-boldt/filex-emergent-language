# Modeling Emergent Lexicon Formation with a Self-Reinforcing Stochastic Process

## Running the Emergent Language System

Create an environment (e.g., using `pip` or `conda`) and install the packages specified in `requirements.txt`.
This code has been tested with Python 3.7, 3.8, and 3.9 on GNU/Linux (3.10 does not work at the time of writing this due to issues with PyTorch).
Since the models used in this project are small, we recommend using a CPU and parallelization rather than a GPU.

The experiments are run by modifying and running `run.sh`.
The `quick_test` configuration takes about 4 minutes in total running on 4 threads on a laptop with an Intel i7-4600U CPU.
See `run.sh` for further documentation.


## Running FiLex (the mathematical model)

The implementation of FiLex can be found in `model-rust`.
It is written in Rust and requires `cargo` to run.
Please see `model-rust/run.sh` for details on how to run the code.
