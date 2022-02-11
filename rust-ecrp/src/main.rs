use rand::prelude::*;
use rand::distributions::Uniform;
use rayon::prelude::*;
use std::fs;
use toml;
use serde_derive::{Serialize, Deserialize};
use std::collections::HashMap;
use std::env;

#[derive(Serialize, Deserialize, Debug, Clone)]
struct Config {
    alpha: f64,
    n_trials: usize,
    n_iters: usize,
    array_size: usize,
    beta_exp_min: f64,
    beta_exp_max: f64,
    sample_method: String,
    process: String,
}

type ConfigFile = HashMap<String, Config>;

type Sampler = fn(& Vec<f64>, f64, &Uniform<f64>, &mut ThreadRng) -> usize;

fn sample_categorical(
    w: & Vec<f64>,
    n: f64,
    dist: & Uniform<f64>,
    rng: &mut ThreadRng
) -> usize {
    let mut x = n * dist.sample(rng);
    let mut i = 0;
    x -= w[i];
    while x > 0.0 {
        i += 1;
        x -= w[i];
    }
    return i;
}

fn sample_softmax(
    w: & Vec<f64>,
    _n: f64,
    dist: & Uniform<f64>,
    rng: &mut ThreadRng
) -> usize {
    let w = w.iter().map(|x| -(-dist.sample(rng).ln()).ln() + x.log2());
    w.enumerate().reduce(|x, y| if x.1 > y.1 { x } else { y }).unwrap().0
}

fn get_entropy(w: & Vec<f64>) -> f64 {
    let n: f64 = w.iter().sum();
    -w.iter()
        .filter(|&&x| x > 0.0)
        .map(|x| (x / n).log2() * x / n).sum::<f64>()
}

// The ECRP should be in some capacity explanatory and in some capacity predictive.

fn ecrp_fixed_buf(cfg: &Config, sample: Sampler, beta: u32) -> f64 {
    // let mut weights: Vec<f64> = Vec::with_capacity(ARRAY_SIZE);
    let scaled_alpha = cfg.alpha / cfg.array_size as f64;
    let mut weights: Vec<f64> = vec![scaled_alpha; cfg.array_size]; 
    let mut addend_idxs: Vec<usize> = Vec::with_capacity(beta as usize);
    addend_idxs.resize(beta as usize, 0);
    let mut rng = rand::thread_rng();
    let dist = Uniform::new(0.0, 1.0);
    for iter in 0..cfg.n_iters {
        for b in 0..beta {
            let idx = sample(&weights, (iter as f64) + cfg.alpha, &dist, &mut rng);
            addend_idxs[b as usize] = idx;
        }
        addend_idxs.iter().for_each(|idx| weights[*idx] += 1.0 / (beta as f64));
    }
    get_entropy(&weights)
}

fn ecrp(cfg: &Config, sample: Sampler, beta: u32) -> f64 {
    let mut weights: Vec<f64> = Vec::with_capacity(cfg.array_size);
    weights.push(0.0);
    let mut addend_idxs: Vec<usize> = Vec::with_capacity(beta as usize);
    addend_idxs.resize(beta as usize, 0);
    let mut rng = rand::thread_rng();
    let dist = Uniform::new(0.0, 1.0);
    for iter in 0..cfg.n_iters {
        let mut weights_len = weights.len();
        if weights[weights_len - 1] == 0.0 {
            weights[weights_len - 1] = cfg.alpha;
        } else {
            weights.push(cfg.alpha);
            weights_len += 1;
        }
        for b in 0..beta {
            let idx = sample(&weights, (iter as f64) + cfg.alpha, &dist, &mut rng);
            addend_idxs[b as usize] = idx;
        }
        weights[weights_len - 1] -= cfg.alpha;
        addend_idxs.iter().for_each(|idx| weights[*idx] += 1.0 / (beta as f64));
    }
    get_entropy(&weights)
}

fn get_config() -> Config {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        panic!("Two arguments required: CONFIG_FILE CONFIG_NAME")
    }
    let config_file = &args[1];
    let config_name = &args[2];
    
    let contents = fs::read_to_string(config_file)
        .expect("Could not read config file.");

    let all_configs: ConfigFile = toml::from_str(&contents)
        .expect("Could not parse config file.");

    format!("Could not find config \"{}\" in config file", config_name);
    all_configs.get(config_name)
        .unwrap_or_else(|| panic!("Could not find config \"{}\" in config file", config_name)).clone()
}

fn main() {
    let cfg = get_config();

    let process_func = match cfg.process.as_str() {
        "ecrp_fixed" => ecrp_fixed_buf,
        "ecrp" => ecrp,
        _ => panic!("Could not find process {}", cfg.process),
    };
    let sampler = match cfg.sample_method.as_str() {
        "categorical" => sample_categorical,
        "gs" => sample_softmax,
        _ => panic!("Could not find sampling method {}", cfg.sample_method),
    };

    let results: Vec<_> = (0..cfg.n_trials).into_par_iter().map(|i| {
    // let results: Vec<_> = (0..N_TRIALS).map(|i| {
        let x = cfg.beta_exp_min + (cfg.beta_exp_max - cfg.beta_exp_min) * (i as f64) / (cfg.n_trials as f64);
        let beta = (10.0_f64).powf(x);
        let beta_int = beta as u32;
        (beta, process_func(&cfg, sampler, beta_int))
    }).collect();
    results.iter().for_each(|(x, y)| {
        println!("{},{}", x, y);
    });
}
