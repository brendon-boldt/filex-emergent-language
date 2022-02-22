mod config;

use rand::distributions::Uniform;
use rand::prelude::*;
use rayon::prelude::*;
use std::env;

/// Sample an index according to a basic categorical distribution.
fn sample_categorical(w: &Vec<f64>, n: f64, dist: &Uniform<f64>, rng: &mut ThreadRng) -> usize {
    let mut x = n * dist.sample(rng);
    let mut i = 0;
    x -= w[i];
    while x > 0.0 {
        i += 1;
        x -= w[i];
    }
    return i;
}

/// Calculate the Shannon entropy (in bits) of a vector of weights
fn get_entropy(w: &Vec<f64>) -> f64 {
    let n: f64 = w.iter().sum();
    -w.iter()
        .filter(|&&x| x > 0.0)
        .map(|x| (x / n).log2() * x / n)
        .sum::<f64>()
}

/// The self-reinforcing stochastic process with a fixed buffer
fn process_fixed_buffer(cfg: &config::Config) -> f64 {
    let scaled_alpha = cfg.alpha / cfg.array_size as f64;
    let mut weights: Vec<f64> = vec![scaled_alpha; cfg.array_size];
    let mut addend_idxs: Vec<usize> = Vec::with_capacity(cfg.beta);
    let mut rng = rand::thread_rng();
    // Since we would have to create a new distribution every time the weights are updated., it is
    // far faster to use a uniform distribution and scale it manually.
    let dist = Uniform::new(0.0, 1.0);
    for iter in 0..cfg.n_iters {
        for b in 0..cfg.beta {
            let idx = sample_categorical(&weights, (iter as f64) + cfg.alpha, &dist, &mut rng);
            addend_idxs[b as usize] = idx;
        }
        addend_idxs
            .iter()
            .for_each(|idx| weights[*idx] += 1.0 / (cfg.beta as f64));
    }
    get_entropy(&weights)
}

/// The Chinese restaurant process with the beta generalization
fn process_infinite_buffer(cfg: &config::Config) -> f64 {
    let mut weights: Vec<f64> = Vec::with_capacity(cfg.array_size);
    weights.push(0.0);
    let mut addend_idxs: Vec<usize> = Vec::with_capacity(cfg.beta);
    let mut rng = rand::thread_rng();
    let dist = Uniform::new(0.0, 1.0);
    let n_iters: u64 = (cfg.n_iters as f64 / cfg.beta as f64) as u64;
    for iter in 0..n_iters {
        let mut weights_len = weights.len();
        if weights[weights_len - 1] == 0.0 {
            weights[weights_len - 1] = cfg.alpha;
        } else {
            weights.push(cfg.alpha);
            weights_len += 1;
        }
        for b in 0..cfg.beta {
            let idx = sample_categorical(&weights, (iter as f64) + cfg.alpha, &dist, &mut rng);
            addend_idxs[b as usize] = idx;
        }
        weights[weights_len - 1] -= cfg.alpha;
        addend_idxs
            .iter()
            .for_each(|idx| weights[*idx] += 1.0 / (cfg.beta as f64));
    }
    get_entropy(&weights)
}

/// Get the command line args.
fn get_config() -> String {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        panic!("One argument is required: CONFIG_NAME");
    }
    return args[1].clone();
}

fn main() {
    let config_name = get_config();
    let configs = config::get_configs(config_name);

    let results = configs.into_par_iter().map(|cfg| {
        let process_func = match cfg.process {
            config::Process::Fixed => process_fixed_buffer,
            config::Process::Base => process_infinite_buffer,
        };
        (cfg.ind_var, process_func(&cfg))
    });

    // Collect the results into a vector to force serialization.
    results.collect::<Vec<_>>().iter().for_each(|(x, y)| {
        println!("{},{}", x, y);
    });
}
